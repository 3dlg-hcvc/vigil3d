"""
preprocess_data.py
------------------
Main script for preprocessing data for specific datasets and methods. Ideally, different methods should have as similar 
preprocessed datasets as possible so as to prevent the amount of storage memory required from exploding, but we may have
to group them into specific classes of methods (e.g. bounding box vs. per-point segmentation methods).
"""

import hydra
import lightning as L
import neptune
from omegaconf import DictConfig

from ovfgvg.data.preprocessing import get_preprocessing_module
from ovfgvg.utils import log_slurm, setup_environment


@hydra.main(version_base=None, config_path="../../config", config_name="data_config")
def main(cfg: DictConfig):
    # 0. load config
    # 1. setup [GPUs]
    setup_environment(cfg.env)

    if cfg.env.logger._target_ == "NeptuneLogger":
        run = neptune.init_run(**cfg.env.logger.params)
    else:
        run = None
    log_slurm(run)

    # 2. load data
    data_module = get_preprocessing_module(cfg.data)

    # 3. preprocess data
    output_folders = data_module.preprocess()

    # 4. log dataset statistics
    # for split in output_folders:
    #     run[f"{split}/scenes"].track_files(output_folders[split])

    if run is not None:
        run.stop()


if __name__ == "__main__":
    main()
