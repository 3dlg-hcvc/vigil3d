from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from .clip_dataset import ClipDataset
from .predictions_dataset import PredictionsLoader as Predictions, predictions_collate_fn
from .toy_dataset import ToyDataset
from ovfgvg.data.types import SceneCollections


__all__ = [
    "Predictions",
    "ClipDataset",
    "ToyDataset",
]


collate_functions = {
    "Predictions": predictions_collate_fn,
}


def get_dataset(dataset: SceneCollections, run_cfg: DictConfig, **kwargs) -> torch.utils.data.Dataset:
    return hydra.utils.instantiate(run_cfg, dataset, **kwargs)


def get_collate_fn(collate_fn):
    return collate_functions[collate_fn]
