"""
evaluate.py
-----------
Main evaluation script for different open-vocabulary, fine-grained methods.
"""

import logging

import hydra
import lightning as L
from omegaconf import DictConfig

from ovfgvg.data.modules import get_datamodule
from ovfgvg.models import get_model, load_from_checkpoint
from ovfgvg.modules import get_lightning_module
from ovfgvg.utils import setup_environment, set_seed, prepare_trainer


@hydra.main(version_base=None, config_path="../../config", config_name="eval_config")
def main(cfg: DictConfig):
    # 1. setup environment
    setup_environment(cfg.env)

    # 2. load model
    # 2a. instantiate model
    model = get_model(cfg.model.model) if hasattr(cfg.model, "model") else None

    eval_model = get_lightning_module(model, cfg.model.eval.module)

    # 2b. preload weights from file/checkpoint
    if cfg.model.eval.load_from_checkpoint:
        eval_model = load_from_checkpoint(eval_model, cfg.model.checkpoint)
    else:
        logging.info("Skipping load from checkpoint.")

    # 3. load data
    datamodule = get_datamodule(cfg.data, cfg.model.eval.data)

    # 4. run model on data and generate metrics or visualizations
    trainer_args = prepare_trainer(cfg.model.eval.trainer)
    trainer = L.Trainer(**trainer_args, default_root_dir=cfg.env.save_dir)
    result = trainer.test(model=eval_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
