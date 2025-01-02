import lightning as L
from omegaconf import DictConfig

from .base import BaseDataModule


def get_datamodule(dataset_config: DictConfig, model_config: DictConfig) -> L.LightningDataModule:
    # dataset_name = dataset_config.dataset_name
    # return datamodule_map[dataset_name](dataset_config, model_config)
    return BaseDataModule(dataset_cfg=dataset_config, run_cfg=model_config)
