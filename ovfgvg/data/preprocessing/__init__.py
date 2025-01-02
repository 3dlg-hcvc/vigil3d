from hydra.utils import instantiate
from omegaconf import DictConfig

from .base import DatasetPreprocessing
from .scanrefer import ScanReferPreprocessing
from .ovfgvg import OVFGVGPreprocessing


preprocessing_modules = {
    mod.name: mod
    for mod in [ScanReferPreprocessing, OVFGVGPreprocessing]
}


def get_preprocessing_module(cfg: DictConfig) -> DatasetPreprocessing:
    return preprocessing_modules[cfg.dataset_name](cfg)
