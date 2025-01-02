from typing import Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.utilities.types import STEP_OUTPUT
from MinkowskiEngine import SparseTensor
import objgraph
from pympler import muppy, summary, tracker

# from ovfgvg.loss import get_loss
# from ovfgvg.metrics import Mask3DMetricsNew
from ovfgvg.utils import compute_label_and_mask, compute_prediction, downsample_label


class ToyModule(L.LightningModule):
    EPSILON = 1e-5

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.loss = get_loss(kwargs["loss"]) if "loss" in kwargs else None
        self.init_lr = kwargs.get("lr")

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inp, label = batch

        out = self(inp)
        loss = self.loss(out, label)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # torch.cuda.empty_cache()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        inp, label = batch

        out = self(inp)
        loss = self.loss(out, label)
        return {"loss": loss}

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # torch.cuda.empty_cache()
        return super().on_validation_batch_end(outputs, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def configure_optimizers(self) -> optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.init_lr)
