# import lightning as L
# from omegaconf import DictConfig
# from torch.utils.data import DataLoader

# from ovfgvg.data.loaders import get_dataset
# from ovfgvg.data.loaders.clip_dataset import clip_collation_fn


# class TextDataModule(L.LightningDataModule):

#     # FIXME: should have a consistent constructor between lightning data modules
#     def __init__(self, text_prompts: list[str], run_cfg: DictConfig):
#         super().__init__()
#         self.text_prompts = text_prompts
#         self.run_cfg = run_cfg

#     def prepare_data(self):
#         pass

#     def setup(self, stage=None):
#         if stage == "fit" or stage is None:
#             self.train_dataset = None
#             self.val_dataset = None

#         if stage == "validate" or stage is None:
#             self.val_dataset = None

#         if stage == "test" or stage is None:
#             self.test_dataset = get_dataset(self.run_cfg, text=self.text_prompts)

#         if stage == "predict" or stage is None:
#             self.predict_dataset = get_dataset(self.run_cfg, text=self.text_prompts)

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(self.train_dataset, batch_size=self.run_cfg.batch_size, num_workers=self.run_cfg.num_workers, collate_fn=clip_collation_fn)

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(self.val_dataset, batch_size=self.run_cfg.batch_size, num_workers=self.run_cfg.num_workers, collate_fn=clip_collation_fn)

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.run_cfg.batch_size,
#             shuffle=False,
#             num_workers=self.run_cfg.num_workers,
#             pin_memory=True,
#             drop_last=False,
#             collate_fn=clip_collation_fn,
#         )

#     def predict_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.predict_dataset,
#             batch_size=self.run_cfg.batch_size,
#             shuffle=False,
#             num_workers=self.run_cfg.num_workers,
#             pin_memory=True,
#             drop_last=False,
#             collate_fn=clip_collation_fn,
#         )
