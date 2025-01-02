import json
import random

import pandas as pd


seed = 31
splits = ["train", "val"]
scanrefer_annotations = "../data/scanrefer/ScanRefer_filtered_{split}.json"
num_annotations = None
# output_name = ".data/datasets/scanrefer/scanrefer_sample.csv"
output_name = ".data/datasets/scanrefer/{split}/metadata.json"
output_mode = "json"

if __name__ == "__main__":
    random.seed(seed)

    vocab = set()
    num_tokens = 0
    for split in splits:
        with open(scanrefer_annotations.format(split=split), "r") as f:
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
                                "ids": [sample["object_id"]],
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
            # with open(output_name.format(split=split), "w") as f:
            #     json.dump(data, f, indent=4)
        elif output_mode == "csv":
            data = []
            for sample_idx in annotation_sample_indices:
                sample = annotations[sample_idx]
                data.append(
                    [sample_idx, sample["scene_id"], sample["object_id"], sample["object_name"], sample["description"]]
                )
            df = pd.DataFrame(data, columns=["idx", "scene_id", "object_id", "object_name", "description"])

            df.to_csv(output_name)

    print(f"Vocab size: {len(vocab)}")
    print(f"{num_tokens=}")