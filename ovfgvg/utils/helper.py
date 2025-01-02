import base64
import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Self, Optional

import numpy as np
import torch
from PIL import Image

this = sys.modules[__name__]


class Null:
    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Null)


def zip_dicts(*args: dict, return_keys: bool = True, default: Any = Null()):
    """
    Zip a list of dictionaries by keys.

    For instance, if the dictionaries are {"a": 1, "b": 2}, {"a": 11, "c": 13}, and {"a": 21, "b": 22, "c": 23}, the
    result would be [("a", [1, 11, 21]), ("b": [2, None, 22]), ("c", [None, 13, 23])] if allow_none is True. Otherwise,
    it is required that all of the dictionaries have the same keys.

    The function will attempt to preserve the order of keys of the first dictionary if no default value is specified.
    However, this may not be guaranteed otherwise, as there is no defined order if some dictionaries are missing certain
    keys.

    :param args: list of dictionaries to merge
    :param return_keys: if True, returns tuples of (key, list of values from each dictionary). Else, returns only list
    of values.
    :param default: if any value other than Null(), allows dictionaries to have different keys and substitutes default
    value where dictionary cannot provide one.
    :yield: tuples of zipped lists for each key
    """

    keys = set()
    allow_none = default != Null()
    for a in args:
        if allow_none:
            keys = keys.union(set(a.keys()))
        elif not keys:  # initially empty
            keys = set(a.keys())
        elif keys != set(a.keys()):  # dictionaries have different keys, and allow_none is False
            raise ValueError("Expected all dictionaries to have the same keys")
    if not allow_none:
        keys = list(a.keys())  # to preserve order when we know the definitive list of keys

    for k in keys:
        if return_keys:
            yield k, [a.get(k, default) if allow_none else a[k] for a in args]
        else:
            yield [a.get(k, default) if allow_none else a[k] for a in args]


def parse_scenes_from_dir(path: str, template: str = "(.*)\.pth") -> dict[str, str | list[str]]:
    """
    Parse scene names from a directory of scene files.

    :param path: path to directory containing scene files
    :param template: regex template to match scene names
    :return: list of scene names
    """

    scenes = {}
    for f in os.listdir(path):
        if re.match(template, f) and os.path.isfile(os.path.join(path, f)):
            scene_id = re.match(template, f).group(1)
            if scene_id in scenes:
                if isinstance(scenes[scene_id], list):
                    scenes[scene_id].append(f)
                else:
                    scenes[scene_id] = [scenes[scene_id], f]
            else:
                scenes[scene_id] = f

    return {
        re.match(template, f).group(1): f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and re.match(template, f)
    }


def rescale(
    data: np.ndarray | torch.Tensor, from_min: float, from_max: float, to_min: float, to_max: float
) -> np.ndarray | torch.Tensor:
    data_norm = (data - from_min) / (from_max - from_min)  # [0, 1]
    return data_norm * (to_max - to_min) + to_min


def adjust_intrinsic(
    intrinsic: np.ndarray, intrinsic_image_dim: list[int | float], image_dim: list[int | float]
) -> np.ndarray:
    """Adjust camera intrinsics based on original and target image dimensions."""

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(np.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def np_to_bytes(tensor: np.array) -> bytes:
    """Convert np.array to bytes. Required specifically for tensorflow application."""
    buff = io.BytesIO()
    img = Image.fromarray(tensor.astype(np.uint8))
    img.save(buff, format="JPEG")
    buff.seek(0)
    return buff.read()


class HierarchicalPath:
    def __init__(self, *paths: str | Path):
        self.paths = paths

    def get(self, path: str | Path):
        for p in self.paths:
            if os.path.exists(os.path.join(p, path)):
                return os.path.join(p, path)

        raise FileNotFoundError(f"File not found in any of the paths: {path}")

    def exists(self, path: str | Path):
        for p in self.paths:
            if os.path.exists(os.path.join(p, path)):
                return True
        return False

    def join(self, path: str | Path) -> Self:
        return HierarchicalPath(*[os.path.join(p, path) for p in self.paths])


def get_batch_counts(batch_indices: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
    """Get counts of points by batch_indices for each point cloud.

    :param batch_indices: a tensor of shape [num_points, ] where each value is the batch index of the corresponding
    point. It is assumed that the points are sorted by batch index.
    :return: _description_
    """
    if batch_size is None:
        batch_size = batch_indices.max().int().item() + 1
    pc_sizes = torch.cat(
        (
            torch.tensor([0], device=batch_indices.device),
            torch.where(batch_indices.diff() == 1)[0] + 1,
            torch.tensor([batch_indices.size(0)], device=batch_indices.device),
        ),
        0,
    )
    return pc_sizes.diff()


def encode_mask(mask: np.ndarray) -> list[list[int]]:
    """Uses RLE encoding"""
    """Find runs of consecutive items in an array."""

    # ensure array is 1D
    assert np.issubdtype(mask.dtype, np.integer)
    mask = mask.flatten()
    length = mask.shape[0]

    # handle empty array
    if length == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(length, dtype=bool)
        loc_run_start[0] = True
        loc_run_start[1:] = np.not_equal(mask[:-1], mask[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = mask[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, length))

        assert run_values.shape[0] == run_lengths.shape[0]
        encoded = np.stack([run_values, run_lengths], axis=1, dtype=np.int64)

        return encoded.tolist()


def decode_mask(
    arr: list[list[int]], dtype=np.uint8, shape: np.ndarray | tuple[int, ...] | list[int] | None = None
) -> np.ndarray:
    arr = np.array(arr)
    size = np.sum(arr[:, 1])
    mask = np.zeros(size, dtype=dtype)

    start = 0
    for value, length in arr:
        mask[start : start + length] = value
        start += length

    if shape is not None:
        mask = mask.reshape(shape)

    return mask


def estimate_num_parameters(model: torch.nn.Module, trainable: bool = False) -> int:
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
