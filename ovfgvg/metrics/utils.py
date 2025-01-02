import torch

from ovfgvg.utils import match_boxes

def is_correct_prediction(preds: torch.tensor, target: torch.tensor, iou: float):
    _, unmatched_pred, unmatched_target = match_boxes(preds, target, iou=iou)

    fp = unmatched_pred.size(0)
    fn = unmatched_target.size(0)
    return fp == 0 and fn == 0