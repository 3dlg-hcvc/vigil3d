# Evaluation

To evaluate your model on ViGiL3D, please follow the below steps to download the requisite data and compare your results against the ground truth. Note that †hese instructions are slightly different than the README, as they leverage a third-party package `torchmetrics-ext` for more convenient evaluation.

## Data Preparation

To run against the full ViGiL3D, you will need to run your model on both ScanNet and ScanNet++ scenes.

ScanNet scenes can be downloaded [here](https://github.com/ScanNet/ScanNet). The ground truth is constructed from the axis-aligned bounding boxes after applying the axis-aligned transform provided in the dataset for each scene (please refer to Multi3DRefer for more details). For final inference, only the reconstructed point cloud or raw RGB-D and pose data should be used.

ScanNet++ scenes can be downloaded [here](https://kaldir.vc.in.tum.de/scannetpp/). The ground truth is constructed from the `segments_anno.json` file per scene, without any additional transformations. For final inference, only the reconstructed point cloud or raw RGB-D and pose data should be used.

Lastly, the descriptions for ViGiL3D can be found [here](https://huggingface.co/datasets/3dlg-hcvc/vigil3d). You can verify that you are working in the correct transformation space by checking your object bounding boxes against the bounding box metadata [here](https://huggingface.co/datasets/torchmetrics-ext/metadata).

## Compute Metrics

Install the latest version of `torchmetrics-ext` for convenient evaluation:
```shell
pip install git+https://github.com/eamonn-zh/torchmetrics_ext
```

Format your predictions in the following form:
```json
{
    "<annotation_id>": [
        [
            [x_min, y_min, z_min],
            [x_max, y_max, z_max]
        ],
        ...
    ]
}
```
where each `annotation_id` is of the form `"{scene_id}_{ann_id}"`, per the Huggingface dataset fields.

Pass the predictions to the dataset per the following example:

```python
import torch
from torchmetrics_ext.metrics.visual_grounding import ViGiL3DMetric

# preds is a dictionary mapping each unique description identifier (formatted as "{scene_id}_{ann_id}")
# to a variable number of predicted axis-aligned bounding boxes in shape (N, 2, 3)
preds = {
    "0a5c013435_03ac3985-6dcf-47e4-98a9-ed10c1775ed9": [
        [
            [0.0, 1.0, 0.5],
            [1.0, 2.0, 1.1]
        ]
    ],
    "scene0050_00_d15e427f-5ba2-4a84-8a41-579a1c856447": [
        [
            [0.0, 1.0, 0.5],
            [1.0, 2.0, 1.1]
        ]
    ],
}

metric = ViGiL3DMetric(split="validation")  # set strict to True to require all descriptions to be included

result = metric(preds)
```

Predictions for both datasets should be included in the same dictionary. To generate metrics for just one of the datasets, evaluate with `strict=False`.
