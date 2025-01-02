import json
import os
import random

import pandas as pd


split = "train"
seed = 31
datasets = ["3RScan", "ARKitScenes", "HM3D", "MultiScan", "ProcThor", "ScanNet", "Structured3D"]
type_ = ["rel2", "relm", "star"]  # rel2, relm, star, chain
post_options = ["gpt", "template"]
file_template = "ssg_ref_{type}_{post}.json"
num_annotations = None
# output_name = f".data/datasets/sceneverse/sceneverse_sample_{type_}.csv"
output_name = f".data/datasets/sceneverse/{split}/metadata.json"
output_mode = "json"  # json or csv

if __name__ == "__main__":
    random.seed(seed)
    vocab = set()
    num_tokens = 0

    data = [] if output_mode == "csv" else {"grounding": []}
    for dataset in datasets:
        if dataset == "ScanNet":
            dataset_dir = f"/datasets/external/sceneverse/download/{dataset}/annotations/refer"
        else:
            dataset_dir = f"/datasets/external/sceneverse/download/{dataset}/{dataset}/annotations"

        for typ in type_:
            for post in post_options:
                inp_path = os.path.join(dataset_dir, file_template.format(type=typ, post=post))
                if os.path.exists(inp_path):
                    with open(inp_path, "r") as f:
                        annotations = json.load(f)

                    if output_mode == "json":
                        for sample_idx in range(len(annotations)):
                            sample = annotations[sample_idx]
                            vocab |= set(sample["utterance"].split())
                            num_tokens += len(sample["utterance"].split())
                            data["grounding"].append(
                                {
                                    "id": sample_idx,
                                    "scene_id": sample["scan_id"],
                                    "text": sample["utterance"],
                                    "entities": [
                                        {
                                            "is_target": True,
                                            "ids": [sample["target_id"]],
                                            "target_name": sample["instance_type"],
                                            "labels": [sample["instance_type"]],
                                            "indexes": None,
                                            "boxes": [],
                                            "metadata": None,
                                            "mask": None,
                                        }
                                    ],
                                }
                            )
                    elif output_mode == "csv":
                        for sample_idx in range(len(annotations)):
                            sample = annotations[sample_idx]
                            data.append(
                                [
                                    sample_idx,
                                    sample["scan_id"],
                                    sample["target_id"],
                                    sample["instance_type"],
                                    sample["utterance"],
                                ]
                            )

    if output_mode == "json":
        with open(output_name, "w") as f:
            json.dump(data, f, indent=4)
        print(f"# prompts: {len(data['grounding'])}")
        print(f"Vocab size: {len(vocab)}")
        print(f"{num_tokens=}")
    elif output_mode == "csv":
        df = pd.DataFrame(data, columns=["idx", "scene_id", "object_id", "object_name", "description"])
        df.to_csv(output_name)
