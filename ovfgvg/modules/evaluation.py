import lightning as L

from ovfgvg.metrics import MetricsSuite


class EvaluationModule(L.LightningModule):
    EPSILON = 1e-5

    def __init__(self, model, **kwargs):
        super().__init__()
        metrics_factory = MetricsSuite.get_factory(cfg=kwargs["metrics"])
        self.test_metrics: MetricsSuite = metrics_factory(prefix="test")

    def test_step(self, batch, batch_idx):
        prediction = batch["predicted_boxes"]
        ground_truth = batch["gt_boxes"]
        result = self.test_metrics(preds=prediction, target=ground_truth, lightning_module=self)
        return {"prediction": prediction, "target": ground_truth, "metrics": result}

    def on_test_epoch_end(self) -> None:
        self.test_metrics.compute(self, reset=True)
        return super().on_test_epoch_end()
