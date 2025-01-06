"""
predictions_dataset.py
---
Dataloader for predictions.

The prediction file must be of the form
[
  {
    "prompt_id": ID of prompt (e.g. "cf49717d-a751-417e-be93-32fa6a4aa1e4"),
    "scene_id": ID of scene (e.g. "scene0012_00"),
    "prompt": Description of scene,
    "predicted_boxes": [
      [
        [
            centroid_x,
            centroid_y,
            centroid_z
        ],
        [
            extent_x,
            extent_y,
            extent_z
        ]
      ]
    ]
  },
  ...
]
"""

import json
from typing import Any, Optional

import torch

from ovfgvg.utils import zip_dicts
from ovfgvg.data.types import SceneCollection


def predictions_collate_fn(batch: list[dict[str, Any]]):
    """
    Note that while each element of the list represents a single batch element, in practice the batch size is
    effectively assumed to be 1.

    :param batch: list of dictionaries containing the following keys:
    {
        "index": index of annotation in dataset,
        "prompt_id": ID of annotation prompt,
        "scene_id": scene_id of annotation,
        "prompt": description of [target] object(s) in scene,
        "predicted_target_ids": list of target_ids per description,
        "predicted_boxes": list of predicted bounding boxes,
        "gt_label": list of corresponding ground truth labels for target,
        "gt_boxes": list of corresponding ground truth bounding boxes,
        "metadata": metadata for prompt,
    }
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

            **cached_data,
    """
    batch_agg = {k: v for k, v in zip_dicts(*batch)}

    return batch_agg


class PredictionsLoader(torch.utils.data.Dataset):
    """Dataloader for predictions."""

    def __init__(
        self,
        dataset: SceneCollection,
        cached_predictions_path: str,
        split="train",
        num_annotations: Optional[int] = None,
        **kw,
    ):
        """
        :param datasets: _description_
        :param voxel_size: _description_, defaults to 0.05
        :param split: _description_, defaults to "train"
        :param aug: _description_, defaults to False
        :param data_aug_color_trans_ratio: _description_, defaults to 0.1
        :param data_aug_color_jitter_std: _description_, defaults to 0.05
        :param data_aug_hue_max: _description_, defaults to 0.5
        :param data_aug_saturation_max: _description, defaults to 0.2
        :param eval_all: if True, returns map of indices from post-voxelization to pre-voxelization. Defaults to False
        :param input_color: if True, returns RGB color features per point on a [-1, 1] scale. Defaults to False
        :param text_features: precomputed text features, defaults to None
        :raises FileNotFoundError: _description_
        """
        super().__init__()
        if split is None:
            split = ""
        self.split = split

        # parse data paths
        self.dataset = dataset
        self.num_annotations = num_annotations

        # load predictions
        try:
            with open(cached_predictions_path, "r") as f:
                predictions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find predictions file: {cached_predictions_path}")
        self.predictions = {}
        for pred in predictions:
            self.predictions[pred["prompt_id"]] = pred

        def filter_(scene_ids: list[str], metadata: list[dict[str, Any]]):
            new_scene_ids = set()
            new_metadata = []
            for prompt in metadata:
                if prompt["id"] in self.predictions:
                    new_scene_ids.add(prompt["scene_id"])
                    new_metadata.append(prompt)
            new_scene_ids = list(new_scene_ids.intersection(set(scene_ids)))
            return new_scene_ids, new_metadata

        # breakpoint()
        self.dataset.apply_filter(filter_)

    def __len__(self):
        if self.num_annotations is not None:
            return min(self.num_annotations, self.dataset.get_num_annotations(self.split))
        else:
            return self.dataset.get_num_annotations(self.split)

    def __getitem__(self, index):
        prompt = self.dataset.get_annotation(self.split, index)
        prediction = self.predictions[prompt.id]

        # FIXME
        boxes = prompt.target.boxes
        gt_boxes = []
        if boxes:
            for box in boxes:
                gt_boxes.append(torch.tensor([box.center.tolist(), box.dimensions.tolist()]))
            gt_boxes = torch.stack(gt_boxes)
        else:
            gt_boxes = torch.zeros(0, 2, 3)
        pred_boxes = torch.tensor(prediction["predicted_boxes"])
        # if pred_boxes.ndim == 2:
        #     pred_boxes = torch.unsqueeze(pred_boxes, 0)

        return {
            "index": index,
            "prompt_id": prompt.id,
            "scene_id": prompt.scene_id,
            "prompt": prompt.text,
            "predicted_target_ids": prediction.get("predicted_ids"),
            "predicted_boxes": pred_boxes,
            "gt_label": prompt.target.labels,
            "gt_boxes": gt_boxes,
            "metadata": prompt.metadata,
        }
