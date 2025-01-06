import json
import logging
import os
from collections import Counter
from enum import Enum, auto
from typing import Any

import numpy as np
import plyfile

from .base import DatasetPreprocessing
from ovfgvg.data.utils import SensorData
from ovfgvg.data.types import Scene, DescriptionPair, ImageView, Entity

from ovfgvg.utils import AbstractProcessWorker, HierarchicalPath


class ProcessMesh(AbstractProcessWorker):
    """Worker for processing a single scene.

    Code credit for a significant portion of this class goes to the original authors of the ScanRefer paper.
    """

    RAW_MESH_FILE = "{scene_id}_vh_clean_2.ply"
    ANNOTATION_FILE = "{scene_id}_vh_clean.aggregation.json"
    SEGMENTATION_FILE = "{scene_id}_vh_clean_2.0.010000.segs.json"
    SENS_FILE = "{scene_id}.sens"
    PROCESSED_FILE = "{scene_id}.pth"
    DEFAULT = -1

    class Status(Enum):
        SUCCESS = auto()
        FAILED = auto()
        SKIPPED = auto()

    def __init__(self, job_queue, response_queue, **kwargs):
        super().__init__(job_queue, response_queue, **kwargs)

        self.annotations = kwargs["annotations"]
        # self.text_embeddings = kwargs["text_embeddings"]
        self.image_size = kwargs["image_size"]
        self.frame_skip = kwargs["frame_skip"]
        self.output_dir = kwargs["output_dir"]
        self.labels = kwargs["labels"]
        self.label_mapping = {label: idx for idx, label in enumerate(self.labels)}

    def process(self, message):
        in_folder, scene_id = message

        scene_path = in_folder.join(scene_id)

        # parse mesh
        scene_mesh = scene_path.get(self.RAW_MESH_FILE.format(scene_id=scene_id))
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

        vertices = np.array([list(x) for x in raw_points.elements[0]])
        coords = np.ascontiguousarray(vertices[:, :3])
        colors = np.ascontiguousarray(vertices[:, 3:6]) / 127.5 - 1  # normalize to -1 to 1; RGB order

        # parse annotations:
        # for each bbox annotation, find the points within the box and assign those the class label
        instance_ids, segmentation_mask = self._get_vertex_to_object_id(
            scene_path.get(self.ANNOTATION_FILE.format(scene_id=scene_id)),
            scene_path.get(self.SEGMENTATION_FILE.format(scene_id=scene_id)),
        )

        try:
            annotations_by_scene = self.annotations["scenes"][scene_id]["annotations"]
        except KeyError:
            logging.error(f"Could not find scene_id in annotations metadata: {scene_id}")
            return ProcessMesh.Status.FAILED, []
        if len(annotations_by_scene) == 0:
            logging.error(f"Scene has no annotations in metadata: {scene_id}")
            return ProcessMesh.Status.FAILED, []

        # text_embedding = np.zeros((len(annotations_by_scene), self.text_embeddings.size(1)), dtype=float)
        descriptions = []
        for idx, annot in enumerate(annotations_by_scene):
            obj_id = annot["object_id"]
            annotation = (instance_ids == obj_id).astype(int)

            # bboxes in the original meshes
            obj_pc = coords[instance_ids == obj_id, 0:3]

            if len(obj_pc) == 0:
                logging.warning(f"Could not match any instances to object IDs for {scene_id=} and {obj_id=}")
                continue

            # Compute axis aligned box
            bbox = Scene.box_from_mask(coords, instance_ids == obj_id)
            entity = Entity(
                is_target=True,
                boxes=[bbox],
                mask=annotation,
                target_name=annot["label"],
                labels=[annot["label"]],
                ids=[obj_id],
            )

            descriptions.append(
                DescriptionPair(
                    id=f"{scene_id}-{idx}", scene_id=scene_id, text=annot["description"], entities=[entity]
                )  # FIXME: should labels have a negative label too?
            )

        # get views
        # views = self._get_views(scene_path, scene_id)

        # scene = Scene(
        #     scene_id=scene_id,
        #     coords=coords,
        #     colors=colors,
        #     views=views,
        # )
        # save output
        # scene.export(os.path.join(self.output_dir, self.PROCESSED_FILE.format(scene_id=scene_id)))
        return ProcessMesh.Status.SUCCESS, descriptions

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
        segmentation_mask = self.DEFAULT * np.ones(shape=(num_verts,), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                segmentation_mask[verts] = self.label_mapping.get(object_ids_to_labels[object_id], self.DEFAULT)

        return instance_ids, segmentation_mask

    def _read_aggregation(self, filename: str):
        object_id_to_segs = {}
        label_to_segs = {}
        object_ids_to_labels = {}
        with open(filename) as f:
            data = json.load(f)
            num_objects = len(data["segGroups"])
            for i in range(num_objects):
                object_id = data["segGroups"][i]["objectId"] + 1  # instance ids should be 1-indexed
                label = data["segGroups"][i]["label"]
                segs = data["segGroups"][i]["segments"]
                object_id_to_segs[object_id] = segs
                object_ids_to_labels[object_id] = label
                if label in label_to_segs:
                    label_to_segs[label].extend(segs)
                else:
                    label_to_segs[label] = segs
        return object_id_to_segs, label_to_segs, object_ids_to_labels

    def _get_views(self, scene_path: HierarchicalPath, scene_id: str) -> list[ImageView]:
        sens_file = scene_path.get(self.SENS_FILE.format(scene_id=scene_id))
        sensor_data = SensorData(sens_file)
        color_frames, intrinsic_color = sensor_data.export_color_images(
            image_size=self.image_size, frame_skip=self.frame_skip
        )
        depth_frames, intrinsic_depth = sensor_data.export_depth_images(
            image_size=self.image_size, frame_skip=self.frame_skip
        )
        poses = sensor_data.export_poses(frame_skip=self.frame_skip)
        views = []
        for color, depth, pose in zip(color_frames, depth_frames, poses):
            views.append(
                ImageView(
                    color=color,
                    depth=depth,
                    pose=pose,
                    intrinsic_color=intrinsic_color,
                    intrinsic_depth=intrinsic_depth,
                )
            )
        return views


class ScanReferPreprocessing(DatasetPreprocessing):
    name = "ScanRefer"

    def get_metadata(self, split: str):
        return os.path.join(self.dataset_cfg.data_root, split, "metadata.json")

    def preprocess(self):
        # TODO: parametrize this by run_cfg, since the prepare_data() may differ from model to model

        raw_data_root = HierarchicalPath(*self.dataset_cfg.preprocess.raw_data_root)
        annotation_folder = self.dataset_cfg.preprocess.raw_annotations
        # scene_collection = SceneCollection(
        #     self.name, self.dataset_cfg.source, "a" if self.dataset_cfg.preprocess.skip_existing else "w"
        # )

        status_counter = Counter()
        for split in self.dataset_cfg.preprocess.splits:
            logging.info(f"Processing split: {split}")
            os.makedirs(os.path.join(self.dataset_cfg.source, split), exist_ok=True)

            # parse annotations and text embeddings
            with open(os.path.join(annotation_folder, f"ScanRefer_filtered_{split}.json"), "r") as f:
                raw_annotations = json.load(f)

            # text_prompts = [annot["description"] for annot in raw_annotations]
            # text_embeddings = self._extract_text_features(text_prompts)
            annotations = self._parse_annotation_file(raw_annotations)

            messages = []
            for scene_id in annotations["scenes"]:
                if not self.dataset_cfg.preprocess.skip_existing or not os.path.exists(
                    os.path.join(self.dataset_cfg.source, split, f"{scene_id}.pth")
                ):
                    if raw_data_root.exists(scene_id):
                        messages.append((raw_data_root, scene_id))
                else:
                    status_counter[ProcessMesh.Status.SKIPPED] += 1

            metadata = {"grounding": [], "scene_metadata": {}}
            for status, descriptions in ProcessMesh.execute_job_generator(
                messages,
                num_workers=self.dataset_cfg.preprocess.num_workers,
                visualize=True,
                annotations=annotations,
                image_size=self.dataset_cfg.preprocess.image_size,
                frame_skip=self.dataset_cfg.preprocess.frame_skip,
                output_dir=os.path.join(self.dataset_cfg.source, split),
                labels=self.dataset_cfg.labels,
            ):
                status_counter[status] += 1
                metadata["grounding"].extend([desc.to_dict() for desc in descriptions])

            with open(os.path.join(self.dataset_cfg.source, split, self.dataset_cfg.metadata), "w") as f:
                json.dump(metadata, f, indent=4)

        # print status tally
        print("PREPROCESSING RESULTS:")
        print(" Status  | Count ")
        print("-----------------")
        print(f" SUCCESS | {status_counter[ProcessMesh.Status.SUCCESS]:>5}")
        print(f" SKIPPED | {status_counter[ProcessMesh.Status.SKIPPED]:>5}")
        print(f" FAILED  | {status_counter[ProcessMesh.Status.FAILED]:>5}")

    def _parse_annotation_file(self, raw_annotations) -> dict[str, list[dict[str, Any]]]:
        annotations = {"num_annotations": len(raw_annotations), "scenes": {}, "annotation_lookup": []}
        for idx, annot in enumerate(raw_annotations):
            if annot["scene_id"] not in annotations["scenes"]:
                annotations["scenes"][annot["scene_id"]] = {"annotations": []}

            # create index to scene_id + relative index mapping for easier lookup later
            annotations["annotation_lookup"].append(
                (annot["scene_id"], len(annotations["scenes"][annot["scene_id"]]["annotations"]))
            )

            # add annotation record
            annotations["scenes"][annot["scene_id"]]["annotations"].append(
                {
                    "id": idx,
                    "rel_id": len(annotations["scenes"][annot["scene_id"]]["annotations"]),
                    "description": annot["description"],
                    "object_id": int(annot["object_id"]),
                    "label": annot["object_name"],
                }
            )
        annotations["num_scenes"] = len(annotations["scenes"])

        return annotations
