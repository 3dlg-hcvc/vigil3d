import json
import random

import pandas as pd

from ovfgvg.utils import makedirs_for_file


seed = 31
split = ["train", "val"]
instruct3d_annotations = "/localhome/atw7/projects/3d-ovfgvg/data/instruct3d/instruct3D_{split}.json"
num_annotations = None
# output_name = f".data/datasets/instruct3d/instruct3d_sample.csv"
output_name = ".data/datasets/instruct3d/{split}/metadata.json"
output_mode = "json"

if __name__ == "__main__":
    random.seed(seed)

    vocab = set()
    num_tokens = 0
    for spl in split:
        with open(instruct3d_annotations.format(split=spl), "r") as f:
            annotations = json.load(f)

        if num_annotations is not None:
            annotation_sample_indices = random.sample(range(len(annotations)), num_annotations)
        else:
            annotation_sample_indices = list(range(len(annotations)))

        if output_mode == "json":
            data = {"grounding": []}
            for sample_idx in annotation_sample_indices:
                sample = annotations[sample_idx]
                vocab |= set(sample["description"].split())
                num_tokens += len(sample["description"].split())
                data["grounding"].append(
                    {
                        "id": sample_idx,
                        "scene_id": sample["scene_id"],
                        "text": sample["description"],
                        "entities": [
                            {
                                "is_target": True,
                                "ids": sample["object_id"],
                                "target_name": sample["object_name"],
                                "labels": [sample["object_name"]],
                                "indexes": None,
                                "boxes": [],
                                "metadata": None,
                                "mask": None,
                            }
                        ],
                    }
                )
            makedirs_for_file(output_name.format(split=spl))
            with open(output_name.format(split=spl), "w") as f:
                json.dump(data, f, indent=4)
        elif output_mode == "csv":
            data = []
            for sample_idx in annotation_sample_indices:
                sample = annotations[sample_idx]
                data.append(
                    [
                        sample_idx,
                        sample["scene_id"],
                        ",".join(map(str, sample["object_id"])),
                        sample["object_name"],
                        sample["description"],
                    ]
                )
            df = pd.DataFrame(data, columns=["idx", "scene_id", "object_id", "object_name", "description"])

            makedirs_for_file(output_name.format(split=spl))
            df.to_csv(output_name.format(split=spl))

    print(f"Vocab size: {len(vocab)}")
    print(f"{num_tokens=}")
