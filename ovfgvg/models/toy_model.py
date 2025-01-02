import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.linear = nn.Linear(cfg.get("dim", 100), 2)

    def forward(self, x):
        return self.linear(x)
