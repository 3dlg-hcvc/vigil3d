from typing import Any, Optional

import matplotlib.pyplot as plt
import torch.nn as nn
import torchmetrics
import lightning as L
from omegaconf import DictConfig


class BaseMetric(torchmetrics.Metric):
    full_state_update: Optional[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_updating = False

    # def forward(self, *args: Any, **kwargs: Any) -> Any:
    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        self.is_updating = True
        out = super()._forward_reduce_state_update(*args, **kwargs)
        self.is_updating = False
        return out


class MetricsSuite(nn.Module):
    """Defines an interface by which to compute a series of metrics at scale.

    This is intended to be a wrapper around other metrics classes, allowing for use in either a lightning setting or
    as a standalone callable."""

    def __init__(self, prefix: str, cfg: DictConfig) -> None:
        super().__init__()
        self.prefix = prefix
        self.on_step = cfg.get("on_step", False)
        self.on_epoch = cfg.get("on_epoch", True)

        self.metrics = torchmetrics.MetricCollection(
            {f"{prefix}/{metric.name}": metric for metric in cfg.metrics},
            compute_groups=cfg.compute_groups,
        )

    def __call__(self, *args, lightning_module: L.LightningModule = None, **kwargs):
        results = self.metrics(*args, **kwargs)
        if lightning_module:
            lightning_module.log_dict(self.metrics, on_step=self.on_step, on_epoch=self.on_epoch)
        return results

    def compute(self, lightning_module: L.LightningModule = None, reset: bool = True) -> Any:
        results = self.metrics.compute()
        if lightning_module:
            lightning_module.log_dict(results, on_step=self.on_step, on_epoch=self.on_epoch)
            for name, metric in self.metrics.items():
                try:
                    fig, _ = metric.plot()
                except NotImplementedError:
                    pass
                else:
                    lightning_module.logger.experiment[name].upload(fig)
                    plt.close(fig)
        if reset:
            self.reset()
        return results

    def finalize(self, lightning_module=None):
        return self(lightning_module=lightning_module)

    def reset(self):
        self.metrics.reset()

    @classmethod
    def get_factory(cls, *args, **kwargs):
        def f_(*a, **kw):
            return cls(*args, *a, **kwargs, **kw)

        return f_
