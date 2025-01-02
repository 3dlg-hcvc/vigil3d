import json
import random
import re

import pandas as pd


seed = 31
splits = ["train", "val"]
threedgrand_annotations = "/datasets/external/3dgrand/grounded_object_reference/{split}_no_grounding.json"
num_annotations = None
# output_name = ".data/datasets/3dgrand/3dgrand_sample.csv"
output_name = ".data/datasets/3dgrand/{split}/metadata.json"
output_mode = "json"  # json or csv


def process_text(text):
    """
    Example input: "This is a <p>wardrobe</p>[<wardrobe-0>] with a warm wooden hue, showcasing a sleek and elongated rectangular form. The <p>wooden wardrobe</p>[<wardrobe-0>] has a smooth and polished wood texture, adding a touch of vintage charm to the living room. The <p>wardrobe</p>[<wardrobe-0>] is positioned behind the <p>grey fabric sofa</p>[<three-seat/multi-seat sofa-1>], <p>coffee tables</p>[<coffee table-2>, <coffee table-3>], and the <p>armchair</p>[<armchair-4>], creating a focal point in the room."
    Example output: "This is a wardrobe with a warm wooden hue, showcasing a sleek and elongated rectangular form. The wooden wardrobe has a smooth and polished wood texture, adding a touch of vintage charm to the living room. The wardrobe is positioned behind the grey fabric sofa, coffee tables, and the armchair, creating a focal point in the room."
    """
    text = re.sub(r"<p>(.*?)<\/p>\[<(.*?)>\]", "\g<1>", text)
    return text


if __name__ == "__main__":
    random.seed(seed)
    num_prompts = 0
    vocab = set()
    num_tokens = 0

    for split in splits:
        with open(threedgrand_annotations.format(split=split), "r") as f:
            annotations = json.load(f)

        if num_annotations is not None:
            annotation_sample_indices = random.sample(range(len(annotations)), num_annotations)
        else:
            annotation_sample_indices = list(range(len(annotations)))

        if output_mode == "json":
            data = {"grounding": []}
            for sample_idx in annotation_sample_indices:
                sample = annotations[sample_idx]
                description = process_text(sample["grounded_object_reference"])
                vocab |= set(description.split())
                num_tokens += len(description.split())
                data["grounding"].append(
                    {
                        "id": sample_idx,
                        "scene_id": sample["scene_id"],
                        "text": description,
                        "entities": [
                            {
                                "is_target": True,
                                "ids": [sample["referred_object_id"]],
                                "target_name": sample["referred_object_text"],
                                "labels": [sample["referred_object_text"]],
                                "indexes": None,
                                "boxes": [],
                                "metadata": None,
                                "mask": None,
                            }
                        ],
                    }
                )
            with open(output_name.format(split=split), "w") as f:
                json.dump(data, f, indent=4)
            num_prompts += len(data["grounding"])

        elif output_mode == "csv":
            data = []
            for sample_idx in annotation_sample_indices:
                sample = annotations[sample_idx]
                data.append(
                    [
                        sample_idx,
                        sample["scene_id"],
                        sample["referred_object_id"],
                        sample["referred_object_text"],
                        sample["grounded_object_reference"],
                    ]
                )
            df = pd.DataFrame(data, columns=["idx", "scene_id", "object_id", "object_name", "description"])

            df.to_csv(output_name.format(split=split))

        else:
            raise ValueError(f"Invalid output mode: {output_mode}")

    print(f"# prompts: {num_prompts}")
    print(f"Vocab size: {len(vocab)}")
    print(f"{num_tokens=}")
