import json
import random
import os

import pandas as pd


seed = 31
directory = "/local-scratch/localhome/atw7/projects/3d-ovfgvg/data/referit3d/"
datasets = ["nr3d", "sr3d+"]
num_annotations = None
# output_name = ".data/datasets/3dgrand/3dgrand_sample.csv"
output_name = ".data/datasets/{dataset}/val/metadata.json"
output_mode = "json"  # json or csv


if __name__ == "__main__":
    random.seed(seed)

    for dataset in datasets:
        num_prompts = 0
        vocab = set()
        num_tokens = 0

        annotations = pd.read_csv(os.path.join(directory, f"{dataset}.csv"))
        breakpoint()

        data = {"grounding": []}
        for idx, sample in enumerate(annotations.itertuples()):
            description = sample.utterance
            vocab |= set(description.split())
            num_tokens += len(description.split())
            data["grounding"].append(
                {
                    "id": idx,
                    "scene_id": sample.scan_id,
                    "text": description,
                    "entities": [
                        {
                            "is_target": True,
                            "ids": [sample.target_id],
                            "target_name": sample.instance_type,
                            "labels": [sample.instance_type],
                            "indexes": None,
                            "boxes": [],
                            "metadata": None,
                            "mask": None,
                        }
                    ],
                }
            )
        with open(output_name.format(dataset=dataset), "w") as f:
            json.dump(data, f, indent=4)
        num_prompts += len(data["grounding"])

        print(f"# prompts: {num_prompts}")
        print(f"Vocab size: {len(vocab)}")
        print(f"{num_tokens=}")
