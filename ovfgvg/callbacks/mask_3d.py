import os
from typing import Any

import lightning as L
import numpy as np
import torch
import seaborn as sns

from ovfgvg.data.visualization import convert_labels_with_palette, export_pointcloud, visualize_labels


class FeatureLogCallback(L.Callback):
    def __init__(self, log_dir, sampling_rate: int = 1):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.sampling_rate = sampling_rate

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx % self.sampling_rate != 0:
            return

        data_path = batch["data_paths"]
        predictions = outputs["predictions"]

        assert len(data_path) == 1
        scene_name = data_path[0].split("/")[-1].split(".pth")[0]
        np.save(
            os.path.join(
                self.log_dir,
                "{}_openscene_feat_{}.npy".format(scene_name, pl_module.feature_type),
            ),
            predictions.cpu().numpy(),
        )


class InputVisualizationCallback(L.Callback):
    def __init__(self, log_dir, sampling_rate: int = 1):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.sampling_rate = sampling_rate

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx % self.sampling_rate != 0:
            return

        data_path = batch["data_paths"]
        pcl = outputs["pcl"]

        assert len(data_path) == 1

        input_color = torch.load(data_path[0])[1]
        export_pointcloud(
            os.path.join(self.log_dir, "{}_input.ply".format(batch_idx)),
            pcl,
            colors=(input_color + 1) / 2,
        )


class PredVisualizationCallback(L.Callback):
    def __init__(
        self, log_dir: str, output_field: str, color_palette: str, suffix: str = "_pred", sampling_rate: int = 1
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.sampling_rate = sampling_rate
        self.output_field = output_field
        self.color_palette = sns.color_palette(color_palette, as_cmap=True)
        self.suffix = suffix

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        return self._generate_visualization(batch, outputs, batch_idx)

    def _generate_visualization(self, batch, outputs: dict[str, Any], batch_idx: int):
        if batch_idx % self.sampling_rate != 0:
            return

        scene_id = outputs["scene_id"]
        coords_reverse = batch["coords"][batch["inds_reverse"], :]
        pcl = coords_reverse[:, 1:].cpu().numpy()

        inds_reverse = batch["inds_reverse"].cpu()
        logits = outputs[self.output_field]
        mask = outputs["mask"][inds_reverse].cpu().numpy()

        # make logits_conf 1-D for a single batch entry
        if len(logits.shape) == 2:
            logits = logits[0, :]

        # map confidence scores to all points
        if logits.shape[0] != batch["coords"].shape[0]:
            logits_all = torch.zeros_like(logits)
            logits_all[mask] = logits
            logits = logits_all[inds_reverse].numpy()
        else:
            logits = logits[inds_reverse].cpu().numpy()

        label_color = self.color_palette(logits)[:, :3].astype(float)
        label_color[mask == 0] = np.array([10.0, 10.0, 10.0]) / 255.0
        export_pointcloud(
            os.path.join(self.log_dir, f"{scene_id}_batch_{batch_idx}{self.suffix}.ply"),
            pcl,
            colors=label_color,
        )


class PredConfVisualizationCallback(L.Callback):
    """Callback visualization for point clouds, with color based on the prediction confidence (scale of 0 to 1)."""

    def __init__(self, log_dir: str, sampling_rate: int = 1, color_palette: str = "crest"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.sampling_rate = sampling_rate
        self.color_palette = sns.color_palette(color_palette, as_cmap=True)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        return self._generate_visualization(batch, outputs, batch_idx)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        return self._generate_visualization(batch, outputs, batch_idx)

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        return self._generate_visualization(batch, outputs, batch_idx)

    def _generate_visualization(self, batch, outputs: dict[str, Any], batch_idx: int):
        if batch_idx % self.sampling_rate != 0:
            return

        scene_id = outputs["scene_id"]
        coords_reverse = batch["coords"][batch["inds_reverse"], :]
        pcl = coords_reverse[:, 1:].cpu().numpy()

        inds_reverse = batch["inds_reverse"].cpu()
        logits_conf = outputs["logits_conf"]
        mask = outputs["mask"][inds_reverse].cpu().numpy()

        # make logits_conf 1-D for a single batch entry
        if len(logits_conf.shape) == 3:
            assert logits_conf.shape[2] == 2, "not binary probability"
            logits_conf = logits_conf[0, :, 1]
        elif len(logits_conf.shape) == 2:
            logits_conf = logits_conf[0, :]

        # map confidence scores to all points
        if logits_conf.shape[0] != batch["coords"].shape[0]:
            logits_conf_all = torch.zeros_like(logits_conf)
            logits_conf_all[mask] = logits_conf
            logits_conf = logits_conf_all[inds_reverse].numpy()
        else:
            logits_conf = logits_conf[inds_reverse].numpy()

        pred_label_color = self.color_palette(logits_conf)[:, :3].astype(float)
        pred_label_color[mask == 0] = np.array([237.0, 102.0, 93.0]) / 255.0
        export_pointcloud(
            os.path.join(self.log_dir, f"{scene_id}_batch_{batch_idx}_conf.ply"),
            pcl,
            colors=pred_label_color,
        )


class GTVisualizationCallback(L.Callback):
    def __init__(self, log_dir, sampling_rate: int = 1):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.sampling_rate = sampling_rate

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx % self.sampling_rate != 0:
            return

        pcl = outputs["pcl"]
        label = batch["labels"]

        # for points not evaluating
        label[label == 255] = len(trainer.datamodule.labelset) - 1
        gt_label_color = convert_labels_with_palette(label.cpu().numpy(), trainer.datamodule.palette)
        export_pointcloud(os.path.join(self.log_dir, "{}_gt.ply".format(batch_idx)), pcl, colors=gt_label_color)
        visualize_labels(
            list(np.unique(label.cpu().numpy())),
            trainer.datamodule.labelset,
            trainer.datamodule.palette,
            os.path.join(self.log_dir, "{}_labels_gt.jpg".format(batch_idx)),
            ncol=5,
        )
