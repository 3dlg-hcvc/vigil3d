import json
import logging
import os
from typing import Optional

import plyfile
import numpy as np

from .base import DatasetUtils
from ovfgvg.data.types import LabeledOrientedBBox


class ScanNetPP(DatasetUtils):
    MESH_FILE = "scans/mesh_aligned_0.05.ply"
    ANNOTATION_FILE = "scans/segments_anno.json"
    SEGMENTS_FILE = "scans/segments.json"

    def __init__(self, scene_dir: str, **kwargs):
        self.scene_dir = scene_dir

    def get_scene_ply_file(self, scene_id: str) -> str:
        return os.path.join(self.scene_dir, scene_id, self.MESH_FILE)

    def load_point_cloud(self, scene_id: str, return_faces: bool = False) -> tuple[np.ndarray, np.ndarray]:
        scene_mesh = self.get_scene_ply_file(scene_id)
        try:
            raw_points = plyfile.PlyData().read(scene_mesh)
        except plyfile.PlyHeaderParseError as e:
            logging.error(f"Could not parse mesh for scene due to corrupt header: {scene_id}")
            raise e
        except plyfile.PlyElementParseError as e:
            logging.error(f"Could not parse mesh for scene due to element parsing error: {scene_id}")
            raise e
        except FileNotFoundError as e:
            logging.error(f"Could not find .ply file: {scene_mesh}")
            raise e

        vertices = np.array([list(x) for x in raw_points.elements[0]])
        coords = np.ascontiguousarray(vertices[:, :3])
        colors = np.ascontiguousarray(vertices[:, 3:6]) / 127.5 - 1  # normalize to -1 to 1; RGB order

        if return_faces:
            faces = np.array([list(x) for x in raw_points.elements[1]])
            faces = faces[:, 0, :]
            return coords, colors, faces

        return coords, colors

    def load_object_boxes(
        self,
        scene_id: str,
        coords: Optional[np.ndarray] = None,
        return_assignment: bool = False,
        return_pc: bool = False,
    ) -> dict[str, LabeledOrientedBBox] | tuple[dict[str, LabeledOrientedBBox], np.ndarray]:
        # load annotations
        scene_path = os.path.join(self.scene_dir, scene_id)
        instance_ids, label_mapping = self._get_vertex_to_object_id(
            os.path.join(scene_path, self.ANNOTATION_FILE),
            os.path.join(scene_path, self.SEGMENTS_FILE),
        )

        def get_box(coords, object_id) -> dict[str, LabeledOrientedBBox]:
            annotation = instance_ids == object_id

            # bboxes in the original meshes
            obj_pc = coords[annotation, 0:3]

            if len(obj_pc) == 0:
                logging.warning(f"Could not match any instances to object IDs for {scene_id=} and {object_id=}")
                return None, None

            # Compute axis aligned box
            return LabeledOrientedBBox.from_mask(object_id, label_mapping[object_id], coords[instance_ids == object_id])

        if coords is None:
            coords, _ = self.load_point_cloud(scene_id)

        boxes = {object_id: get_box(coords, object_id) for object_id in label_mapping}

        outputs = [boxes]
        if not return_assignment and not return_pc:
            return outputs[0]
        else:
            if return_assignment:
                if coords is None:
                    coords, _ = self.load_point_cloud(scene_id)
                assignments = np.ones(coords.shape[0], dtype=int) * -1
                for object_id, box in boxes.items():
                    assignments[box.contains(coords)] = object_id
                outputs.append(assignments)
            if return_pc:
                outputs.append(coords)
            return tuple(outputs)

    def load_object_mask(self, scene_id: str, coords: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[int, str]]:
        scene_path = os.path.join(self.scene_dir, scene_id)
        with open(os.path.join(scene_path, self.ANNOTATION_FILE), "r") as f:
            annotations = json.load(f)

        with open(os.path.join(scene_path, self.SEGMENTS_FILE), "r") as f:
            segments_data = json.load(f)
        segments = np.array(segments_data["segIndices"])

        if coords is None:
            coords, _ = self.load_point_cloud(scene_id)

        object_label_mapping = {}
        object_mask = np.ones(coords.shape[0], dtype=int) * -1
        for obj in annotations["segGroups"]:
            object_id = obj["objectId"]
            label = obj["label"]
            segments_obj = obj["segments"]

            # vertices
            v_mask = np.isin(segments, segments_obj)
            object_mask[v_mask] = object_id
            object_label_mapping[object_id] = label

        return object_mask, object_label_mapping

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