from omegaconf import DictConfig

import torch


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, **kwargs):
        self.dim = kwargs.get("dim", 100)
        self.length = kwargs.get("length", 100)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        inp = torch.rand((self.dim,), dtype=torch.float)
        label = 1 if (inp > 0.5).sum() > self.dim / 2 else 0
        return inp, label
