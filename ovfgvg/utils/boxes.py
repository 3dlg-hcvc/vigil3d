"""
boxes.py
--------
Helper functions related to the processing of bounding boxes.
"""

import torch
from scipy.optimize import linear_sum_assignment


def compute_iou(box_1: torch.Tensor, box_2: torch.Tensor):
    centroid_1 = box_1[0, :]
    extent_1 = box_1[1, :]
    centroid_2 = box_2[:, 0, :]
    extent_2 = box_2[:, 1, :]

    box_min_1 = centroid_1 - extent_1 / 2
    box_max_1 = centroid_1 + extent_1 / 2
    box_min_2 = centroid_2 - extent_2 / 2
    box_max_2 = centroid_2 + extent_2 / 2

    # Compute the intersection volume
    intersection_min = torch.maximum(box_min_1, box_min_2)
    intersection_max = torch.minimum(box_max_1, box_max_2)
    intersect_dims = torch.maximum(intersection_max - intersection_min, torch.zeros_like(intersection_max))
    intersect = torch.prod(intersect_dims)

    union = torch.prod(box_max_1 - box_min_1) + torch.prod(box_max_2 - box_min_2, dim=1) - intersect

    return torch.where(union > 0, intersect / union, torch.zeros((box_2.size(0),), device=box_2.device))


def match_boxes(predicted: torch.Tensor, target: torch.Tensor, iou=0.25):
    """
    Match predicted bounding boxes to ground truth bounding boxes based on IoU.

    :param predicted: list of predicted bounding boxes
    :param target: list of ground truth bounding boxes
    :param iou: IoU threshold for matching

    :return: list of matched indices
    """
    # assumes exactly one box
    if predicted.ndim == 2:
        predicted = torch.unsqueeze(predicted, 0)

    # both are empty
    if predicted.size(0) == 0 and target.size(0) == 0:
        return (
            torch.empty((0, 2), device=target.device),
            torch.empty((0,), device=target.device),
            torch.empty((0,), device=target.device),
        )

    # prediction is empty
    elif predicted.size(0) == 0:
        return (
            torch.empty((0, 2), device=target.device),
            torch.empty((0,), device=target.device),
            torch.arange(target.size(0), device=target.device),
        )

    # target is empty
    elif target.size(0) == 0:
        return (
            torch.empty((0, 2), device=target.device),
            torch.arange(predicted.size(0), device=target.device),
            torch.empty((0,), device=target.device),
        )

    iou_matrix = torch.zeros(predicted.size(0), target.size(0))
    for i, pred in enumerate(predicted):
        iou_matrix[i, :] = compute_iou(pred, target)

    pred_ind, target_ind = linear_sum_assignment(iou_matrix, maximize=True)
    iou_matchings = iou_matrix[pred_ind, target_ind]
    pred_mask = iou_matchings >= iou
    pred_ind = torch.from_numpy(pred_ind)[pred_mask]
    target_ind = torch.from_numpy(target_ind)[pred_mask]

    unmatched_pred = torch.tensor([i for i in range(len(predicted)) if i not in pred_ind], device=target.device)
    unmatched_target = torch.tensor([i for i in range(len(target)) if i not in target_ind], device=target.device)
    matches = torch.stack((pred_ind, target_ind), dim=1).to(target.device)
    return matches, unmatched_pred, unmatched_target
