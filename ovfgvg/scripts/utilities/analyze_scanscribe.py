import json
import random

import pandas as pd


split = "train"
seed = 31
scanscribe_annotations = [
    "/datasets/external/scanscribe/gpt_gen_language.json",
    "/datasets/external/scanscribe/template_gen_language.json",
]
num_annotations = None
# output_name = ".data/datasets/scanscribe/scanscribe_sample.csv"
output_name = f".data/datasets/scanscribe/{split}/metadata_full.json"
output_mode = "json"  # json or csv

if __name__ == "__main__":
    random.seed(seed)
    vocab = set()
    num_tokens = 0
    if output_mode == "json":
        data = {"grounding": []}
    else:
        data = []

    for annot_file in scanscribe_annotations:
        with open(annot_file, "r") as f:
            annotations = json.load(f)

        if num_annotations is not None:
            annotation_sample_indices = random.sample(range(len(annotations)), num_annotations)
        else:
            annotation_sample_indices = list(range(len(annotations)))

        if output_mode == "json":
            for sample_idx in annotation_sample_indices:
                sample = annotations[sample_idx]
                vocab |= set(sample["sentence"].split())
                num_tokens += len(sample["sentence"].split())
                data["grounding"].append(
                    {
                        "id": sample_idx,
                        "scene_id": sample["scan_id"],
                        "text": sample["sentence"],
                        "entities": [],
                    }
                )
        elif output_mode == "csv":
            for sample_idx in annotation_sample_indices:
                sample = annotations[sample_idx]
                data.append(
                    [
                        sample_idx,
                        sample["scan_id"],
                        sample["sentence"],
                    ]
                )

    if output_mode == "json":
        with open(output_name, "w") as f:
            json.dump(data, f, indent=4)
    elif output_mode == "csv":
        df = pd.DataFrame(data, columns=["idx", "scene_id", "description"])

        df.to_csv(output_name)

    print(f"Vocab size: {len(vocab)}")
    print(f"{num_tokens=}")
