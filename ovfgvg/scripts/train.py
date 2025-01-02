"""
train.py
--------
Main training script for different open-vocabulary, fine-grained methods.
"""

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from ovfgvg.data.modules import get_datamodule
from ovfgvg.models import get_model, load_from_checkpoint
from ovfgvg.modules import get_lightning_module
from ovfgvg.utils import setup_environment, prepare_trainer


@hydra.main(version_base=None, config_path="../../config", config_name="train_config")
def main(cfg: DictConfig):
    # 1. setup environment
    setup_environment(cfg.env)

    # 2. load model
    # 2a. instantiate model
    model = get_model(cfg.model.model)
    train_model = get_lightning_module(model, cfg.model.train.module)

    # 2b. preload weights from file/checkpoint
    if cfg.model.train.load_from_checkpoint:
        train_model = load_from_checkpoint(train_model, cfg.model.checkpoint)

    # 3. load data
    datamodule = get_datamodule(cfg.data, cfg.model.train.data)

    # 4. run model on data and generate metrics or visualizations
    trainer_args = prepare_trainer(cfg.model.train.trainer)
    trainer = L.Trainer(**trainer_args, default_root_dir=cfg.env.save_dir)
    trainer.fit(model=train_model, datamodule=datamodule, ckpt_path=cfg.ckpt)


if __name__ == "__main__":
    main()
