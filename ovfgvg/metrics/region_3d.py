"""
region_3d.py
------------
Metrics related to evaluating performance of a bounding box output for visual grounding.
"""

from typing import Any

import torch

from ovfgvg.metrics.base import BaseMetric as Metric
from ovfgvg.metrics.utils import is_correct_prediction
from ovfgvg.utils import match_boxes


class AccuracyAtIoU(Metric):
    """Computes the accuracy at a given IoU threshold for a mask prediction over points in a point cloud.

    It is assumed that each prediction is a binary mask over all or a subset of the points in the point cloud. To
    recover the original class label for that visual grounding prompt, we use the annotation_metadata JSON.

    Args:
        iou_threshold: IoU threshold to use for the accuracy calculation
    """

    def __init__(self, iou: float, plottable: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.iou_threshold = iou
        self.plottable = plottable

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @property
    def name(self):
        return f"acc@iou{int(self.iou_threshold * 100):02d}"

    @property
    def internal(self):
        # return {"correct": self.correct, "total": self.total}
        return [["correct", self.correct], ["total", self.total]]

    def update(self, preds: list[torch.Tensor], target: list[torch.Tensor]) -> None:
        # FIXME: only works with batch size 1
        assert len(target) == 1
        preds = preds[0]
        target = target[0]
        correct = 1 if is_correct_prediction(preds, target, iou=self.iou_threshold) else 0

        self.correct += correct
        self.total += 1

        return correct

    def compute(self) -> torch.Tensor:
        if self.is_updating:
            return self.internal
        else:
            return self.aggregate(**dict(self.internal))

    @classmethod
    def aggregate(cls, correct, total, **_):
        return correct / total

    def reset(self) -> None:
        super().reset()


class F1ScoreAtIoU(Metric):
    """Computes the F1 score at a given IoU threshold for a mask prediction over points in a point cloud.

    It is assumed that each prediction is a binary mask over all or a subset of the points in the point cloud. To
    recover the original class label for that visual grounding prompt, we use the annotation_metadata JSON.

    Args:
        iou_threshold: IoU threshold to use for the accuracy calculation
    """

    def __init__(self, iou: float, plottable: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.iou_threshold = iou
        self.plottable = plottable

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    @property
    def name(self):
        return f"f1@iou{int(self.iou_threshold * 100):02d}"

    @property
    def internal(self):
        # return {"tp": self.tp, "fp": self.fp, "fn": self.fn}
        return [["tp", self.tp], ["fp", self.fp], ["fn", self.fn]]

    def update(self, preds: list[torch.Tensor], target: list[torch.Tensor]) -> None:
        # FIXME: only works with batch size 1
        assert len(target) == 1
        preds = preds[0]
        target = target[0]
        matches, unmatched_pred, unmatched_target = match_boxes(preds, target, iou=self.iou_threshold)

        tp = matches.size(0)
        fp = unmatched_pred.size(0)
        fn = unmatched_target.size(0)

        self.tp += tp
        self.fp += fp
        self.fn += fn

        return torch.tensor([tp, fp, fn])

    def compute(self) -> torch.Tensor:
        if self.is_updating:
            return self.internal
        else:
            return self.aggregate(**dict(self.internal))

    @classmethod
    def aggregate(cls, tp, fp, fn, **_):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    def reset(self) -> None:
        super().reset()
