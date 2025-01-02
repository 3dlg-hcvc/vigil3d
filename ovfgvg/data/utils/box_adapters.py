import os
from typing import Optional

import numpy as np
import torch

from ovfgvg.data.types import LabeledOrientedBBox


class ClusteringMixin:
    def __init__(self, epsilon: float = 0.05, min_samples: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.min_samples = min_samples

    def load_object_boxes(
        self,
        scene_id: str,
        coords: Optional[np.ndarray] = None,
        return_assignment: bool = False,
        return_pc: bool = False,
    ) -> dict[str, LabeledOrientedBBox] | tuple[dict[str, LabeledOrientedBBox], np.ndarray]:
        if coords is None:
            coords, _ = self.load_point_cloud(scene_id)

        boxes, assignments = LabeledOrientedBBox.cluster_boxes(coords, self.epsilon, self.min_samples)

        box_mapping = {box.id: box for box in boxes}

        outputs = [box_mapping]
        if not return_assignment and not return_pc:
            return outputs[0]
        else:
            if return_assignment:
                outputs.append(assignments)
            if return_pc:
                outputs.append(coords)
            return tuple(outputs)


class Mask3dMixin:
    MASK3D_FILE_FORMAT = "{scene_id}.pt"

    def __init__(self, mask3d_directory: str, **kwargs):
        super().__init__(**kwargs)
        self.mask3d_directory = mask3d_directory

    def load_object_boxes(
        self,
        scene_id: str,
        coords: Optional[np.ndarray] = None,
        return_assignment: bool = False,
        return_pc: bool = False,
    ) -> dict[str, LabeledOrientedBBox] | tuple[dict[str, LabeledOrientedBBox], np.ndarray]:
        if coords is None:
            coords, _ = self.load_point_cloud(scene_id)

        instances = torch.load(os.path.join(self.mask3d_directory, self.MASK3D_FILE_FORMAT.format(scene_id=scene_id)))

        box_mapping = {}
        assignments = np.ones(coords.shape[0], dtype=int) * -1
        for idx, obj_instance in enumerate(instances):
            inst_mask = obj_instance["segments"]
            pcd = coords[inst_mask]
            label = obj_instance["label"]

            box_mapping[idx] = LabeledOrientedBBox.from_mask(idx, label, pcd)
            assignments[inst_mask] = idx

        outputs = [box_mapping]
        if not return_assignment and not return_pc:
            return outputs[0]
        else:
            if return_assignment:
                outputs.append(assignments)
            if return_pc:
                outputs.append(coords)
            return tuple(outputs)
