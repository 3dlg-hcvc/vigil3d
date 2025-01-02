import json
import os

import click
import pandas as pd


@click.command()
@click.option("--dataset", help="path to dataset csv")
@click.option("--output", help="path to output folder")
def analyze_dataset(dataset, output):
    data = pd.read_csv(dataset)
    stats = {}

    # analyze number of scenes and prompts per scenes
    stats["num_prompts"] = len(data)
    num_prompts_per_scene = data.scene_id.value_counts()
    stats["num_scenes"] = len(num_prompts_per_scene)
    stats["num_prompts_per_scene"] = {"mean": num_prompts_per_scene.mean(), "std": num_prompts_per_scene.std()}
    prompt_lengths = data.prompt.apply(lambda x: len(x.split()))
    stats["prompt_lengths"] = {"mean": prompt_lengths.mean(), "std": prompt_lengths.std()}

    # get distribution of target objects
    target_objects = data.target_label.value_counts()
    stats["num_target_classes"] = len(target_objects)
    stats["target_object_distribution"] = target_objects.to_dict()

    # get statistics for each metadata field
    metadata_fields = [
        c for c in data.columns if c not in {"scene_id", "prompt", "prompt_id", "target_id", "target_label"}
    ]
    stats["metadata"] = {}
    for field in metadata_fields:
        value_prop = data[field].value_counts() / stats["num_prompts"]
        stats["metadata"][field] = value_prop.to_dict()

    with open(os.path.join(output, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    analyze_dataset()
