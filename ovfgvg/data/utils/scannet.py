# code supports usages in Python3.
# Adapted from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py

import json
import logging
import os
import struct
from typing import Optional

import plyfile
import numpy as np
import zlib
import imageio
import cv2

from .base import DatasetUtils
from ovfgvg.data.types import LabeledOrientedBBox

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class ScanNet(DatasetUtils):
    MESH_FILE_FORMAT = "{scene_id}_vh_clean_2.ply"
    ANNOTATION_FILE_FORMAT = "{scene_id}_vh_clean.aggregation.json"
    SEGMENTATION_FILE_FORMAT = "{scene_id}_vh_clean_2.0.010000.segs.json"

    def __init__(self, scene_dir: str, **kwargs):
        self.scene_dir = scene_dir

    def get_scene_ply_file(self, scene_id: str) -> str:
        return os.path.join(self.scene_dir, scene_id, self.MESH_FILE_FORMAT.format(scene_id=scene_id))

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
        # for each bbox annotation, find the points within the box and assign those the class label
        scene_path = os.path.join(self.scene_dir, scene_id)
        instance_ids, label_mapping = self._get_vertex_to_object_id(
            os.path.join(scene_path, self.ANNOTATION_FILE_FORMAT.format(scene_id=scene_id)),
            os.path.join(scene_path, self.SEGMENTATION_FILE_FORMAT.format(scene_id=scene_id)),
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

        box_mapping = {object_id: get_box(coords, object_id) for object_id in label_mapping}
        outputs = [box_mapping]
        if not return_assignment and not return_pc:
            return outputs[0]
        else:
            if return_assignment:
                outputs.append(instance_ids)
            if return_pc:
                outputs.append(coords)
            return tuple(outputs)

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


class RGBDFrame:

    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(
            4, 4
        )
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(struct.unpack("c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b"".join(struct.unpack("c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise ValueError("invalid type")

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise ValueError("invalid type")

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:

    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path=None, image_size=None, frame_skip=1):
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            logging.debug("exporting", len(self.frames) // frame_skip, " depth frames to", output_path)

        output_frames = []
        original_size = None
        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:
                if f == 0:  # only for first image
                    original_size = (depth.shape[1], depth.shape[0])  # (w, h)
                depth = cv2.resize(depth, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)

            if output_path is not None:
                imageio.imwrite(os.path.join(output_path, str(f) + ".png"), depth)
            output_frames.append(depth)

        # transform intrinsic
        intrinsic = self.intrinsic_depth.copy()
        intrinsic[0, 0] *= float(image_size[0]) / float(original_size[0])
        intrinsic[1, 1] *= float(image_size[1]) / float(original_size[1])
        intrinsic[0, 2] *= float(image_size[0] - 1) / float(original_size[0] - 1)
        intrinsic[1, 2] *= float(image_size[1] - 1) / float(original_size[1] - 1)
        return output_frames, intrinsic

    def export_color_images(self, output_path=None, image_size=None, frame_skip=1):
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            logging.debug("exporting", len(self.frames) // frame_skip, "color frames to", output_path)

        output_frames = []
        original_size = None
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                if f == 0:  # only for first image
                    original_size = (color.shape[1], color.shape[0])  # (w, h)

                color = cv2.resize(color, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)

            if output_path is not None:
                imageio.imwrite(os.path.join(output_path, str(f) + ".jpg"), color)
            output_frames.append(color)

        # transform intrinsic
        intrinsic = self.intrinsic_color.copy()
        intrinsic[0, 0] *= float(image_size[0]) / float(original_size[0])
        intrinsic[1, 1] *= float(image_size[1]) / float(original_size[1])
        intrinsic[0, 2] *= float(image_size[0] - 1) / float(original_size[0] - 1)
        intrinsic[1, 2] *= float(image_size[1] - 1) / float(original_size[1] - 1)

        return output_frames, intrinsic

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path=None, frame_skip=1):
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            logging.debug("exporting", len(self.frames) // frame_skip, "camera poses to", output_path)

        output_poses = []
        for f in range(0, len(self.frames), frame_skip):
            if output_path is not None:
                self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + ".txt"))
            output_poses.append(self.frames[f].camera_to_world)

        return output_poses

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logging.debug("exporting camera intrinsics to", output_path)
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt"))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt"))
        self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt"))
        self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt"))
