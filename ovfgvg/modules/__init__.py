import hydra
import lightning as L
import torch.nn as nn
from omegaconf import DictConfig

from .evaluation import EvaluationModule
from .text import TextEncoderModule
from .toy import ToyModule

__all__ = [
    "OpenSceneModule",
    "LERFModule",
    "TextEncoderModule",
    "OVFGVG2DModule",
    "OVFGVG25DModule",
    "OVFGVG3DModule",
    "OVFGVGTextModule",
    "OpenSegModule",
    "ToyModule",
    "EvaluationModule",
]


def get_lightning_module(model: nn.Module, module_config: DictConfig) -> type[L.LightningModule]:
    return hydra.utils.instantiate(module_config, model)
