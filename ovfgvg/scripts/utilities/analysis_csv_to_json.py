"""
A one-off script to convert the dataset analysis CSV to a JSON:
prompt_id,scene_id,object_id,object_name,description
201,scene0025_00,4,couch,this is a small dark sofa; the sofa looks black. it is placed in the corner of a room.

JSON format:
{
    "grounding": [
        {
            "id": "7997234f-37b6-453f-abdd-1b22ab7ec230",
            "scene_id": "scene0139_00",
            "text": "This is the orange object labeled \"bounce\" on one of the washing machines.",
            "entities": [
                {
                    "is_target": true,
                    "ids": [
                        3
                    ],
                    "target_name": "dryer sheets",
                    "labels": [
                        "dryer sheets"
                    ]
                }
            ]
        }
    ]
}
"""

import json

import click
import pandas as pd

input_csv = ".data/datasets/analysis/val/dataset_analysis_prompts.csv"


@click.command()
@click.option("--csv", "-i", "input_csv", help="path to manual dataset analysis CSV")
@click.option("--output", help="path to output")
def parse(input_csv, output):
    prompts_raw = pd.read_csv(input_csv)

    metadata = {"grounding": []}
    for prompt in prompts_raw.itertuples(index=False):
        metadata["grounding"].append(
            {
                "id": prompt.prompt_id,
                "scene_id": prompt.scene_id,
                "text": prompt.description,
                "entities": [
                    {
                        "is_target": True,
                        "ids": str(prompt.object_id).split(","),
                        "target_name": prompt.object_name,
                        "labels": [prompt.object_name],
                        "boxes": [],
                    }
                ],
                "metadata": [],
            }
        )

    with open(output, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    parse()
