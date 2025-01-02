import json
import logging
import os
import traceback
from collections import Counter
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
import plyfile

from .base import DatasetPreprocessing
from ovfgvg.data.types import Scene, OrientedBBox, DescriptionPair, Entity, LabeledOrientedBBox
from ovfgvg.utils import AbstractProcessWorker, HierarchicalPath


class ProcessMesh(AbstractProcessWorker):
    RAW_MESH_FILE = "scans/mesh_aligned_0.05.ply"
    ANNOTATION_FILE = "scans/segments_anno.json"
    PROCESSED_FILE = "{scene_id}.pth"
    DEFAULT = -1

    class Status(Enum):
        SUCCESS = auto()
        FAILED = auto()
        SKIPPED = auto()

    def __init__(self, job_queue, response_queue, **kwargs):
        super().__init__(job_queue, response_queue, **kwargs)

        self.annotations = kwargs["annotations"]
        self.image_size = kwargs["image_size"]
        self.frame_skip = kwargs["frame_skip"]
        self.output_dir = kwargs["output_dir"]
        self.mesh_file_format = kwargs.get("mesh_file_format", self.RAW_MESH_FILE)
        self.annotation_file_format = kwargs.get("annotation_file_format", self.ANNOTATION_FILE)

    def _load_boxes(self, scene_path, scene_id) -> dict[str, LabeledOrientedBBox]:
        raise NotImplementedError

    @staticmethod
    def parse_targets(targets: str):
        t_id_list = targets.split(",")
        target_ids = []
        for t in t_id_list:
            try:
                target_ids.append(int(t))
            except ValueError:
                pass
        return target_ids

    def process(self, message):
        in_folder, scene_id = message

        scene_path = in_folder.join(scene_id)

        # parse mesh
        scene_mesh = scene_path.get(self.mesh_file_format.format(scene_id=scene_id))
        try:
            raw_points = plyfile.PlyData().read(scene_mesh)
        except plyfile.PlyHeaderParseError:
            logging.error(f"Could not parse mesh for scene due to corrupt header: {scene_id}")
            return ProcessMesh.Status.FAILED, []
        except plyfile.PlyElementParseError:
            logging.error(f"Could not parse mesh for scene due to element parsing error: {scene_id}")
            return ProcessMesh.Status.FAILED, []
        except FileNotFoundError:
            logging.error(f"Could not find .ply file: {scene_mesh}")
            return ProcessMesh.Status.FAILED, []

        try:
            vertices = np.array([list(x) for x in raw_points.elements[0]])
            coords = np.ascontiguousarray(vertices[:, :3])
            colors = np.ascontiguousarray(vertices[:, 3:6]) / 127.5 - 1  # normalize to -1 to 1; RGB order
            object_bboxes = self._load_boxes(scene_path, scene_id, coords)

            # parse annotations:
            # for each bbox annotation, find the points within the box and assign those the class label
            annotations_by_scene = self.annotations[self.annotations["scene_id"] == scene_id]

            # text_embedding = np.zeros((len(annotations_by_scene), self.text_embeddings.size(1)), dtype=float)
            descriptions = []
            for annot in annotations_by_scene.itertuples(index=False, name="Prompt"):
                # Compute axis aligned box
                bboxes = []
                target_ids = []
                labels = []
                if annot.object_id:
                    target_ids = self.parse_targets(str(annot.object_id))
                    for t_id in target_ids:
                        box: LabeledOrientedBBox = object_bboxes[t_id]
                        labels.append(box.label)
                        bboxes.append(box.get_box())

                entity = Entity(
                    is_target=True, boxes=bboxes, ids=target_ids, labels=labels, target_name=annot.object_name
                )

                metadata = {
                    name: value
                    for name, value in annot._asdict().items()
                    if name
                    not in {
                        "scene_id",
                        "description",
                        "prompt_id",
                        "object_id",
                        "object_name",
                    }
                }

                descriptions.append(
                    DescriptionPair(
                        id=annot.prompt_id,
                        scene_id=scene_id,
                        text=annot.description,
                        entities=[entity],
                        metadata=metadata,
                    )
                )

            # get views
            scene = Scene(
                scene_id=scene_id,
                coords=coords,
                boxes=list(object_bboxes.values()),
                colors=colors,
            )
            # save output
            scene.export(os.path.join(self.output_dir, self.PROCESSED_FILE.format(scene_id=scene_id)))
            return ProcessMesh.Status.SUCCESS, descriptions
        except Exception as e:
            traceback.print_exc()
            return ProcessMesh.Status.FAILED, []


class ProcessMeshScanNetPP(ProcessMesh):
    SEGMENTATION_FILE = "segments.json"

    def __init__(self, job_queue, response_queue, **kwargs):
        super().__init__(job_queue, response_queue, **kwargs)

        self.segmentation_file_format = kwargs.get("segmentation_file_format", self.SEGMENTATION_FILE)

    def _load_boxes(self, scene_path, scene_id, coords) -> dict[str, LabeledOrientedBBox]:
        # load annotations
        # for each bbox annotation, find the points within the box and assign those the class label
        instance_ids, label_mapping = self._get_vertex_to_object_id(
            scene_path.get(self.annotation_file_format.format(scene_id=scene_id)),
            scene_path.get(self.segmentation_file_format.format(scene_id=scene_id)),
        )

        def get_box(object_id):
            annotation = instance_ids == object_id

            # bboxes in the original meshes
            obj_pc = coords[annotation, 0:3]

            if len(obj_pc) == 0:
                logging.warning(f"Could not match any instances to object IDs for {scene_id=} and {object_id=}")
                return None

            # Compute axis aligned box
            return Scene.box_from_mask(coords, instance_ids == object_id)

        return {
            object_id: LabeledOrientedBBox.from_box(
                id=object_id, label=label_mapping[object_id], box=get_box(object_id)
            )
            for object_id in label_mapping
        }

    def _get_vertex_to_object_id(self, annotation_filename: str, segmentation_filename: str):
        # get mapping of object_ids to vertices in segmentation
        object_id_to_segs, _, object_ids_to_labels = self._read_aggregation(annotation_filename)

        # load segmentation file mapping
        seg_to_verts = {}
        with open(segmentation_filename) as f:
            data = json.load(f)
            num_verts = len(data["segIndices"])
            for i in range(num_verts):
                seg_id = data["segIndices"][i]
                if seg_id in seg_to_verts:
                    seg_to_verts[seg_id].append(i)
                else:
                    seg_to_verts[seg_id] = [i]

        # generate mapping of point indexes to object IDs
        instance_ids = np.zeros(shape=(num_verts,), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id

        return instance_ids, object_ids_to_labels

    def _read_aggregation(self, filename: str):
        object_id_to_segs = {}
        label_to_segs = {}
        object_ids_to_labels = {}
        with open(filename) as f:
            data = json.load(f)
            num_objects = len(data["segGroups"])
            for i in range(num_objects):
                object_id = data["segGroups"][i]["objectId"]
                label = data["segGroups"][i]["label"]
                segs = data["segGroups"][i]["segments"]
                object_id_to_segs[object_id] = segs
                object_ids_to_labels[object_id] = label
                if label in label_to_segs:
                    label_to_segs[label].extend(segs)
                else:
                    label_to_segs[label] = segs
        return object_id_to_segs, label_to_segs, object_ids_to_labels

    @staticmethod
    def get_scene_ids(path, splits):
        scene_ids = set()
        if len(splits) > 0:
            for split in splits:
                split_file = path.format(split=split)
                with open(split_file, "r") as f:
                    scene_ids_by_split = set([line[:-1] for line in f.readlines()])
                    scene_ids = scene_ids.union(scene_ids_by_split)
        return scene_ids


class ProcessMeshScanNet(ProcessMesh):
    SEGMENTATION_FILE = "{scene_id}_vh_clean.aggregation.json"

    def __init__(self, job_queue, response_queue, **kwargs):
        super().__init__(job_queue, response_queue, **kwargs)

        self.segmentation_file_format = kwargs.get("segmentation_file_format", self.SEGMENTATION_FILE)

    def _load_boxes(self, scene_path, scene_id, coords) -> dict[str, LabeledOrientedBBox]:
        # load annotations
        # for each bbox annotation, find the points within the box and assign those the class label
        instance_ids, label_mapping = self._get_vertex_to_object_id(
            scene_path.get(self.annotation_file_format.format(scene_id=scene_id)),
            scene_path.get(self.segmentation_file_format.format(scene_id=scene_id)),
        )

        def get_box(object_id):
            annotation = instance_ids == object_id

            # bboxes in the original meshes
            obj_pc = coords[annotation, 0:3]

            if len(obj_pc) == 0:
                logging.warning(f"Could not match any instances to object IDs for {scene_id=} and {object_id=}")
                return None

            # Compute axis aligned box
            return Scene.box_from_mask(coords, instance_ids == object_id)

        return {
            object_id: LabeledOrientedBBox.from_box(
                id=object_id, label=label_mapping[object_id], box=get_box(object_id)
            )
            for object_id in label_mapping
        }

    def _get_vertex_to_object_id(self, annotation_filename: str, segmentation_filename: str):
        # get mapping of object_ids to vertices in segmentation
        object_id_to_segs, _, object_ids_to_labels = self._read_aggregation(annotation_filename)

        # load segmentation file mapping
        seg_to_verts = {}
        with open(segmentation_filename) as f:
            data = json.load(f)
            num_verts = len(data["segIndices"])
            for i in range(num_verts):
                seg_id = data["segIndices"][i]
                if seg_id in seg_to_verts:
                    seg_to_verts[seg_id].append(i)
                else:
                    seg_to_verts[seg_id] = [i]

        # generate mapping of point indexes to object IDs
        instance_ids = np.zeros(shape=(num_verts,), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id

        return instance_ids, object_ids_to_labels

    def _read_aggregation(self, filename: str):
        object_id_to_segs = {}
        label_to_segs = {}
        object_ids_to_labels = {}
        with open(filename) as f:
            data = json.load(f)
            num_objects = len(data["segGroups"])
            for i in range(num_objects):
                object_id = data["segGroups"][i]["objectId"]
                label = data["segGroups"][i]["label"]
                segs = data["segGroups"][i]["segments"]
                object_id_to_segs[object_id] = segs
                object_ids_to_labels[object_id] = label
                if label in label_to_segs:
                    label_to_segs[label].extend(segs)
                else:
                    label_to_segs[label] = segs
        return object_id_to_segs, label_to_segs, object_ids_to_labels

    @staticmethod
    def get_scene_ids(path, splits):
        scene_ids = set()
        if len(splits) > 0:
            for split in splits:
                split_file = path.format(split=split)
                with open(split_file, "r") as f:
                    scene_ids_by_split = set(json.load(f))
                    scene_ids = scene_ids.union(scene_ids_by_split)
        return scene_ids


process_classes = {"ScanNetPP": ProcessMeshScanNetPP, "ScanNet": ProcessMeshScanNet}


class OVFGVGPreprocessing(DatasetPreprocessing):
    name = "ViGiL3D"

    def preprocess(self):
        process_class = process_classes[self.dataset_cfg.preprocess.process_class]
        # load splits
        scene_ids = process_class.get_scene_ids(
            self.dataset_cfg.preprocess.split_path, self.dataset_cfg.preprocess.input_splits
        )

        # load annotations
        raw_annotations = pd.read_csv(self.dataset_cfg.preprocess.raw_annotations, dtype={"target_id": str})
        raw_annotations.fillna(0, inplace=True)
        scene_ids = scene_ids.intersection(set(raw_annotations["scene_id"].tolist()))
        raw_data_root = HierarchicalPath(os.path.join(self.dataset_cfg.preprocess.raw_data_root))

        status_counter = Counter()
        os.makedirs(os.path.join(self.dataset_cfg.source, self.dataset_cfg.preprocess.split), exist_ok=True)

        messages = []
        for scene_id in scene_ids:
            if not self.dataset_cfg.preprocess.skip_existing or not os.path.exists(
                os.path.join(self.dataset_cfg.source, self.dataset_cfg.preprocess.split, f"{scene_id}.pth")
            ):
                if raw_data_root.exists(scene_id):
                    messages.append((raw_data_root, scene_id))
            else:
                status_counter[ProcessMesh.Status.SKIPPED] += 1

        metadata = {"grounding": [], "scene_metadata": {}}
        for status, descriptions in process_class.execute_job_generator(
            messages,
            num_workers=self.dataset_cfg.preprocess.num_workers,
            visualize=True,
            annotations=raw_annotations,
            image_size=self.dataset_cfg.preprocess.image_size,
            frame_skip=self.dataset_cfg.preprocess.frame_skip,
            mesh_file_format=self.dataset_cfg.preprocess.mesh_file_format,
            annotation_file_format=self.dataset_cfg.preprocess.annotation_file_format,
            segmentation_file_format=self.dataset_cfg.preprocess.get("segmentation_file_format"),
            output_dir=os.path.join(self.dataset_cfg.source, self.dataset_cfg.preprocess.split),
        ):
            status_counter[status] += 1
            metadata["grounding"].extend([desc.to_dict() for desc in descriptions])

        # print status tally
        print("PREPROCESSING RESULTS:")
        print(" Status  | Count ")
        print("-----------------")
        print(f" SUCCESS | {status_counter[ProcessMesh.Status.SUCCESS]:>5}")
        print(f" SKIPPED | {status_counter[ProcessMesh.Status.SKIPPED]:>5}")
        print(f" FAILED  | {status_counter[ProcessMesh.Status.FAILED]:>5}")

        with open(
            os.path.join(self.dataset_cfg.source, self.dataset_cfg.preprocess.split, self.dataset_cfg.metadata), "w"
        ) as f:
            json.dump(metadata, f, indent=4)
