import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ovfgvg.data.dataset import get_dataset, get_collate_fn
from ovfgvg.data.types import SceneCollection, SceneCollections


class BaseDataModule(L.LightningDataModule):
    """
    Data module for loading all datasets.

    We use a generic data module rather than one specific to a dataset because we need it to support multiple datasets.
    """

    def __init__(self, dataset_cfg: DictConfig, run_cfg: DictConfig):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.run_cfg = run_cfg

    @property
    def name(self):
        return self.dataset_cfg.dataset_name

    @property
    def dataset_type(self):
        return self.dataset_cfg.dataset_type

    def setup(self, stage=None):
        if hasattr(self.dataset_cfg, "dataset_name") and hasattr(self.dataset_cfg, "source"):
            filter_f = instantiate(self.dataset_cfg.filter) if hasattr(self.dataset_cfg, "filter") else None
            datasets = [SceneCollection(self.dataset_cfg.dataset_name, self.dataset_cfg.source, filter_=filter_f)]
            dataset = SceneCollections(datasets)
        else:
            dataset = None

        if stage == "fit" or stage is None:
            self.train_dataset = get_dataset(
                dataset, self.run_cfg.dataset.train, split=self.run_cfg.split.train
            )
            self.val_dataset = get_dataset(
                dataset, self.run_cfg.dataset.val, split=self.run_cfg.split.val
            )

        if stage == "test" or stage is None:
            self.test_dataset = get_dataset(
                dataset, self.run_cfg.dataset, split=self.run_cfg.split.test
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = get_dataset(
                dataset, self.run_cfg.dataset, split=self.run_cfg.split.test
            )  # TODO: could use a separate 'predict' split instead

    def train_dataloader(self) -> DataLoader:
        params = OmegaConf.to_container(self.run_cfg.dataloader.train)
        if params.get("collate_fn") is not None:
            params["collate_fn"] = get_collate_fn(params["collate_fn"])
        return DataLoader(self.train_dataset, **params)

    def val_dataloader(self) -> DataLoader:
        params = OmegaConf.to_container(self.run_cfg.dataloader.val)
        if params.get("collate_fn") is not None:
            params["collate_fn"] = get_collate_fn(params["collate_fn"])
        return DataLoader(self.val_dataset, **params)

    def test_dataloader(self) -> DataLoader:
        params = OmegaConf.to_container(self.run_cfg.dataloader.test)
        if params.get("collate_fn") is not None:
            params["collate_fn"] = get_collate_fn(params["collate_fn"])
        return DataLoader(self.test_dataset, **params)

    def predict_dataloader(self) -> DataLoader:
        params = OmegaConf.to_container(self.run_cfg.dataloader.test)
        if params.get("collate_fn") is not None:
            params["collate_fn"] = get_collate_fn(params["collate_fn"])
        return DataLoader(self.predict_dataset, **params)
