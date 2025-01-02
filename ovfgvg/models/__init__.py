import logging
import os

import hydra
import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .clip_ import get_clip as CLIP
from .mask3d import Mask3D
# from .openscene import DisNet as OpenScene
# from .openseg import get_openseg as OpenSeg
# from .ovfgvg import OVFGVG
from .toy_model import ToyModel


def get_model(model_config: DictConfig) -> nn.Module:
    return hydra.utils.instantiate(model_config)


def load_from_checkpoint(
    pl_module: L.LightningModule, checkpoint_path: dict[str, str] | DictConfig
) -> L.LightningModule:
    if len(checkpoint_path) == 0:  # no-op
        return pl_module

    match checkpoint_path["type"]:
        case "checkpoint":  # lightning checkpoint
            if not os.path.exists(checkpoint_path["value"]):
                logging.warning(f"Checkpoint not found: {checkpoint_path['value']}. Training from scratch.")
                return pl_module

            checkpoint = torch.load(checkpoint_path["value"], map_location="cpu")
            pl_module.load_state_dict(checkpoint["state_dict"], strict=checkpoint_path["strict"])
            logging.info(f"Loaded checkpoint: {checkpoint_path['value']}")
        case "local_weights":
            if not os.path.exists(checkpoint_path["value"]):
                logging.warning(f"Checkpoint not found: {checkpoint_path['value']}. Training from scratch.")
                return pl_module

            weights = torch.load(checkpoint_path["value"], map_location="cpu")

            # iterate through keys to get to actual model weights
            for key in checkpoint_path.get("key", []):
                weights = weights[key]

            pl_module.model.load_state_dict(weights, strict=checkpoint_path["strict"])
            logging.info(f"Loaded checkpoint: {checkpoint_path['value']}")
        case _:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_path['type']}")
    return pl_module
