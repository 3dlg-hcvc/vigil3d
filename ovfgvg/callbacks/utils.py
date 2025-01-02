import os
from datetime import datetime
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig


class OutputLogCallback(L.Callback):
    def __init__(self, log_dir, cfg: dict | DictConfig, output_name: str, format: str = "pth"):
        self.log_dir = log_dir
        self.output_name = output_name
        self.format = format
        self.cfg = cfg
        os.makedirs(log_dir, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, Any],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        outputs["date_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outputs["config"] = self.cfg

        if self.format == "pth":
            torch.save(
                outputs,
                os.path.join(self.log_dir, self.output_name.format(**outputs)),
            )
        else:
            raise NotImplementedError(f"Unsupported format: {self.format}")
