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
from ovfgvg.utils import setup_environment, predict


@hydra.main(version_base=None, config_path="../../config", config_name="inference_config")
def main(cfg: DictConfig):
    # 0. load config
    # 1. setup [GPUs]
    setup_environment(cfg.env)
    infer_config = cfg.model.predict if "predict" in cfg.model else cfg.model.eval

    # 2. load model
    model = get_model(cfg.model.model)
    predict_model = get_lightning_module(model, infer_config.module)

    # 2b. preload weights from file/checkpoint
    if cfg.model.eval.load_from_checkpoint and "checkpoint" in cfg.model:
        predict_model = load_from_checkpoint(predict_model, cfg.model.checkpoint)
    else:
        logging.info("Skipping load from checkpoint.")

    # 3. load data
    datamodule = get_datamodule(cfg.data, infer_config.data)

    # 4. run model on data
    # 5. generate metrics or visualizations
    predict(predict_model, datamodule, infer_config.trainer, cfg.env.save_dir)


if __name__ == "__main__":
    main()
