from typing import Optional

import numpy as np

from ovfgvg.data.types import LabeledOrientedBBox


class DatasetUtils:
    def get_scene_ply_file(self, scene_id: str) -> str:
        raise NotImplementedError

    def load_point_cloud(self, scene_id: str, return_faces: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Load point cloud for a given scene.

        :param scene_id: ID of scene
        :return: tuple of (coords, color)
        """
        raise NotImplementedError

    def load_object_boxes(
        self, scene_id: str, coords: Optional[np.ndarray] = None, return_assignment: bool = False
    ) -> dict[str, LabeledOrientedBBox] | tuple[dict[str, LabeledOrientedBBox], np.ndarray]:
        """
        Load ground truth bounding boxes for a given scene.

        :param scene_id: ID of scene
        :return: dict mapping object IDs to their ground truth bounding boxes
        """
        raise NotImplementedError

    def load_object_mask(self, scene_id: str, coords: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[int, str]]:
        """
        Load object mask for a given scene.

        :param scene_id: ID of scene
        :return: tuple of (mask, object_label_mapping)
        """
        raise NotImplementedError
