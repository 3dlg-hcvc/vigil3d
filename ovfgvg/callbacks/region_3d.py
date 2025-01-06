import itertools
import json
import os
import numpy as np
from typing import Optional

import lightning as L
import pandas as pd
import torch
from tqdm import tqdm

from ovfgvg.metrics import statistics
from ovfgvg.scripts.analyze import Metrics
from ovfgvg.utils import AbstractProcessWorker


class MetricsCallback(L.Callback):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = {}
        for metric, value in trainer.callback_metrics.items():
            metrics[metric] = value.cpu().numpy().tolist()
        with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        return super().on_test_epoch_end(trainer, pl_module)


class SubgroupAnalysisCallback(L.Callback):
    METADATA_PREFIX = "<md>"
    METRICS_PREFIX = "<mtr>"

    def __init__(
        self,
        log_dir: str,
        stat_test_name: str = "two_proportion_z_test",
        stat_metric: str = "test/acc@iou50",
        latex_columns: Optional[str] = None,
        latex_metric: Optional[str] = None,
    ):
        self.predictions = []
        self.log_dir = log_dir
        self.statistical_test = getattr(statistics, stat_test_name)
        self.stat_metric = f"<mtr>/{stat_metric}"
        self.latex_columns = latex_columns
        self.latex_metric = latex_metric
        os.makedirs(log_dir, exist_ok=True)

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.predictions = []  # reset callback
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, dict[str, torch.Tensor]],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        prompt_ids = batch["prompt_id"]
        scene_ids = batch["scene_id"]

        for i in range(len(prompt_ids)):
            if batch["metadata"][i] is not None:
                metadata_values = {f"{self.METADATA_PREFIX}/{k}": v for k, v in batch["metadata"][i].items()}
            else:
                metadata_values = {}
            self.predictions.append(
                {
                    "prompt_id": prompt_ids[i],
                    "scene_id": scene_ids[i],
                    "prediction": outputs["prediction"][i].cpu().tolist(),
                    "target": outputs["target"][i].cpu().tolist(),
                    **metadata_values,
                    **{
                        # f"{self.METRICS_PREFIX}/{k}": v[i].item() if len(prompt_ids) > 1 else v.item()
                        f"{self.METRICS_PREFIX}/{k}": dict(v[i]) if len(prompt_ids) > 1 else dict(v)
                        for k, v in outputs["metrics"].items()
                    },
                }
            )

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        def aggregate(data, metric_col):
            metric_parse = metric_col.split("/")
            split = metric_parse[1]
            metric = getattr(pl_module, f"{split}_metrics").metrics["/".join(metric_parse[1:])]

            agg_state = None
            for state in data[metric_col]:
                if agg_state is None:
                    agg_state = state.copy()
                else:
                    for k in agg_state:  # TODO: add other options for aggregating state
                        agg_state[k] = agg_state[k] + state[k]
            if agg_state is None:
                raise ValueError("Received an empty subgroup dataframe.")
            return metric.aggregate(**agg_state).item()

        data = pd.DataFrame.from_records(self.predictions)
        data.to_csv(os.path.join(self.log_dir, "predictions.csv"))

        scores = []
        metrics_cols = [col for col in data.columns if col.startswith(self.METRICS_PREFIX)]
        for col in data.columns:
            if col.startswith(self.METADATA_PREFIX):
                for value in data[col].unique():
                    filtered_data = data[data[col] == value]
                    scores.append(
                        {
                            "param": col,
                            "value": value,
                            "num_prompts": len(filtered_data),
                            **{metric_col: aggregate(filtered_data, metric_col) for metric_col in metrics_cols},
                        }
                    )
        subgroup_data = pd.DataFrame.from_records(scores)

        if subgroup_data.empty:
            return super().on_test_epoch_end(trainer, pl_module)

        # generate latex table row
        if self.latex_columns and self.latex_metric:
            try:
                overall_metric = trainer.callback_metrics[self.latex_metric]
            except KeyError as e:
                raise KeyError(f"Could not find metric in callbacks: {self.latex_metric}")

            pref_latex_metric = f"<mtr>/{self.latex_metric}"
            latex_output = [overall_metric]  # TODO: get method name?
            subgroup_params = subgroup_data["param"].unique()
            for col, value in self.latex_columns:
                if f"<md>/{col}" not in subgroup_params:
                    raise ValueError(f"Could not find column '<md>/{col}' in subgroups for LaTeX table generation.")

                subgroup_by_param = subgroup_data[subgroup_data["param"] == f"<md>/{col}"]
                metrics = subgroup_by_param[subgroup_by_param["value"] == value]
                if len(metrics) == 0:
                    raise ValueError(
                        f"Could not find value for corresponding param in subgroups for LaTeX table generation: {col=}, {value=}"
                    )
                latex_output.append(metrics[pref_latex_metric].item())
            latex_output_str = " & ".join(map(lambda x: f"{x * 100:.1f}", latex_output)) + " \\\\"
        else:
            latex_output_str = None

        # evaluate significance
        significant_params = []
        for data_by_param in subgroup_data.groupby("param"):
            for value in self.evaluate_statistical_significance(data_by_param[1], self.stat_metric):
                try:
                    value = int(value)
                except ValueError:
                    pass
                significant_params.append((data_by_param[0], value))

        # save results
        if latex_output_str:
            with open(os.path.join(self.log_dir, "table_row.tex"), "w") as f:
                f.write(latex_output_str)
        subgroup_data.to_csv(os.path.join(self.log_dir, "subgroup_analysis.csv"))
        with open(os.path.join(self.log_dir, "significant_params.json"), "w") as f:
            json.dump(significant_params, f)

        return super().on_test_epoch_end(trainer, pl_module)

    def evaluate_statistical_significance(self, data: pd.DataFrame, metric_col: str) -> float:
        values = data["value"].unique()

        metrics_with_significance = []
        for value in values:
            data_value = data[data["value"] == value]
            size_1 = data_value["num_prompts"].item()
            score_1 = data_value[metric_col].item()

            other = data[data["value"] != value]
            size_2 = other["num_prompts"].sum()
            score_2 = other.apply(lambda x: x["num_prompts"] * x[metric_col], axis=1).sum() / size_2

            p_value = self.statistical_test(size_1, size_2, score_1, score_2)
            if p_value < 0.05:
                metrics_with_significance.append(value)

        return metrics_with_significance


class ReWeightWorker(AbstractProcessWorker):
    def __init__(self, job_queue, response_queue, **kwargs):
        super().__init__(job_queue, response_queue, **kwargs)
        self.possible_values = kwargs["possible_values"]
        self.weights_and_metrics = kwargs["weights_and_metrics"]

    def process(self, message):
        combinations = message
        partial_reweighted_accuracy = 0
        for combination in combinations:
            weights = [
                self.weights_and_metrics[(metric.partition("/")[2], value)][0]
                for metric, value in zip(self.possible_values.keys(), combination)
            ]
            metrics = [
                self.weights_and_metrics[(metric.partition("/")[2], value)][1]
                for metric, value in zip(self.possible_values.keys(), combination)
            ]
            partial_reweighted_accuracy += np.prod(weights) * np.mean(metrics)
        return partial_reweighted_accuracy


class ReWeightedAccuracyCallback(L.Callback):
    METADATA_PREFIX = "<md>"
    METRICS_PREFIX = "<mtr>"

    METRICS_MAPPING = {
        ("<md>/granularity", "generic"): "m_percent_target_ref_generic",
        ("<md>/granularity", "categorical"): "m_percent_target_ref_categorical",
        ("<md>/granularity", "fine-grained"): "m_percent_target_ref_fine_grained",
        "<md>/num_target_not_first_np": "m_percent_target_not_first_np",
        "<md>/num_coreferences": "m_percent_coreferences",
        "<md>/num_negation": "m_percent_negation",
        "<md>/num_attribute_type_style": "m_percent_attribute_type_style",
        "<md>/num_attribute_type_texture": "m_percent_attribute_type_texture",
        "<md>/num_attribute_type_function": "m_percent_attribute_type_function",
        "<md>/num_attribute_type_state": "m_percent_attribute_type_state",
        "<md>/num_attribute_type_text_label": "m_percent_attribute_type_text_label",
        "<md>/num_attribute_type_number": "m_percent_attribute_type_number",
        "<md>/num_attribute_type_shape": "m_percent_attribute_type_shape",
        "<md>/num_attribute_type_color": "m_percent_attribute_type_color",
        "<md>/num_attribute_type_size": "m_percent_attribute_type_size",
        "<md>/num_attribute_type_material": "m_percent_attribute_type_material",
        "<md>/num_relationship_type_contain": "m_percent_relationship_type_contain",
        "<md>/num_relationship_type_near": "m_percent_relationship_type_near",
        "<md>/num_relationship_type_vertical": "m_percent_relationship_type_vertical",
        "<md>/num_relationship_type_arrangement": "m_percent_relationship_type_arrangement",
        "<md>/num_relationship_type_comparison": "m_percent_relationship_type_comparison",
        "<md>/num_relationship_type_viewpoint_dependent": "m_percent_relationship_type_viewpoint_dependent",
        "<md>/num_relationship_type_ordinal": "m_percent_relationship_type_ordinal",
        "<md>/num_relationship_type_far": "m_percent_relationship_type_far",
        "<md>/num_anchor_type_single": "m_percent_anchor_type_single",
        "<md>/num_anchor_type_multiple": "m_percent_anchor_type_multiple",
        "<md>/num_anchor_type_non_object": "m_percent_anchor_type_non_object",
        "<md>/num_anchor_type_viewpoint": "m_percent_anchor_type_viewpoint",
    }

    def __init__(
        self,
        log_dir: str,
        analysis_file: str,
        metric: str,
        mode: str,
        num_samples: int = 100_000,
    ):
        self.predictions = []
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metric = metric
        self.mode = mode  # "approximate" or "stochastic" or "exact"
        with open(analysis_file, "r") as f:
            results_cache = json.load(f)
            self.analysis = Metrics.from_json(results_cache.get("metrics", {}))
        self.num_samples = num_samples

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.predictions = []  # reset callback
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: dict[str, dict[str, torch.Tensor]],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        prompt_ids = batch["prompt_id"]
        scene_ids = batch["scene_id"]

        for i in range(len(prompt_ids)):
            if batch["metadata"][i] is not None:
                metadata_values = {f"{self.METADATA_PREFIX}/{k}": v for k, v in batch["metadata"][i].items()}
            else:
                metadata_values = {}
            self.predictions.append(
                {
                    "prompt_id": prompt_ids[i],
                    "scene_id": scene_ids[i],
                    "prediction": outputs["prediction"][i].cpu().tolist(),
                    "target": outputs["target"][i].cpu().tolist(),
                    **metadata_values,
                    **{
                        # f"{self.METRICS_PREFIX}/{k}": v[i].item() if len(prompt_ids) > 1 else v.item()
                        f"{self.METRICS_PREFIX}/{k}": dict(v[i]) if len(prompt_ids) > 1 else dict(v)
                        for k, v in outputs["metrics"].items()
                    },
                }
            )

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        def aggregate(data, metric_col):
            metric_parse = metric_col.split("/")
            split = metric_parse[1]
            metric = getattr(pl_module, f"{split}_metrics").metrics["/".join(metric_parse[1:])]

            agg_state = None
            for state in data[metric_col]:
                if agg_state is None:
                    agg_state = state.copy()
                else:
                    for k in agg_state:  # TODO: add other options for aggregating state
                        agg_state[k] = agg_state[k] + state[k]
            if agg_state is None:
                raise ValueError("Received an empty subgroup dataframe.")
            return metric.aggregate(**agg_state).item()

        data = pd.DataFrame.from_records(self.predictions)

        scores = []
        metrics_cols = [col for col in data.columns if col.startswith(self.METRICS_PREFIX)]
        for col in data.columns:
            if col.startswith(self.METADATA_PREFIX):
                for value in data[col].unique():
                    filtered_data = data[data[col] == value]
                    scores.append(
                        {
                            "param": col,
                            "value": value,
                            "num_prompts": len(filtered_data),
                            **{metric_col: aggregate(filtered_data, metric_col) for metric_col in metrics_cols},
                        }
                    )
        subgroup_data = pd.DataFrame.from_records(scores)

        if subgroup_data.empty:
            return super().on_test_epoch_end(trainer, pl_module)

        # compute reweighted accuracy
        weights_and_metrics = {}
        possible_values = {}
        for _, row in subgroup_data.iterrows():
            metric = row.param
            value = row.value
            if metric in {
                "<md>/pair_id",
                "<md>/is_valid",
                "<md>/num_attributes_target",
                "<md>/num_attributes_anchors",
                "<md>/num_relationships_target",
                "<md>/num_relationships_anchors",
            }:
                continue

            possible_values[metric] = subgroup_data[subgroup_data["param"] == metric]["value"].unique()

            # parse out <md> prefix
            try:
                analysis_name = self.METRICS_MAPPING[metric]
            except KeyError:
                analysis_name = self.METRICS_MAPPING[(metric, value)]
            metric_name = metric.partition("/")[2]
            dataset_proportion = (
                getattr(self.analysis, analysis_name)
                if value != 0
                else 1 - getattr(self.analysis, analysis_name)
            )
            weights_and_metrics[(metric_name, value)] = [
                dataset_proportion,
                row[f"<mtr>/{self.metric}"],
            ]

        most_common = sorted([(metric, value) for (metric, value) in weights_and_metrics if value != 0], key=lambda x: weights_and_metrics[x][0], reverse=True)
        print(most_common[:5])

        if self.mode == "approximate":
            reweighted_accuracy = 0
            combinations = list(itertools.product(*possible_values.values()))
            for combination in tqdm(combinations):
                weights = [
                    weights_and_metrics[(metric.partition("/")[2], value)][0]
                    for metric, value in zip(possible_values.keys(), combination)
                ]
                metrics = [
                    weights_and_metrics[(metric.partition("/")[2], value)][1]
                    for metric, value in zip(possible_values.keys(), combination)
                ]
                reweighted_accuracy += np.prod(weights) * np.mean(metrics)
        
        elif self.mode == "stochastic":
            reweighted_accuracy = 0
            for _ in tqdm(range(self.num_samples)):
                metrics = []
                for metric in possible_values:
                    metric_name = metric.partition("/")[2]
                    prob = np.array([weights_and_metrics[(metric_name, value)][0] for value in
                    possible_values[metric]])
                    prob = prob / np.sum(prob)
                    value = np.random.choice(
                        possible_values[metric],
                        p=prob,
                    )
                    # if value != 0.0:
                    metrics.append(weights_and_metrics[(metric_name, value)][1])
                reweighted_accuracy += 1 if np.random.rand() < np.mean(metrics) else 0
            reweighted_accuracy /= self.num_samples

        else:
            raise NotImplementedError

        output = {
            "metric": self.metric,
            "mode": self.mode,
            "reweighted_accuracy": reweighted_accuracy,
        }

        with open(os.path.join(self.log_dir, "reweighted_accuracy.json"), "w") as f:
            json.dump(output, f, indent=4)

        return super().on_test_epoch_end(trainer, pl_module)
