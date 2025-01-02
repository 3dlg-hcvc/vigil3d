from omegaconf import DictConfig


class DatasetPreprocessing:
    IGNORE = 255

    def __init__(self, dataset_cfg: DictConfig):
        self.dataset_cfg = dataset_cfg

    def preprocess(self):
        raise NotImplementedError
