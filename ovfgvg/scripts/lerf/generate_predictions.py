"""
The code for this baseline is adapted from https://github.com/sled-group/chat-with-nerf, who similarly used the LERF
baseline in their paper for LLM-Grounder.
"""

import json
import os

import click
import clip
import h5py
import numpy as np
import torch
from tqdm import tqdm

from ovfgvg.data.types import SceneCollection
from ovfgvg.data.utils import get_dataset


def assign_clusters(objects, pred):
    num_prompts = pred.shape[1]
    sim_scores = np.zeros((len(objects), num_prompts))
    for idx, (_, indices) in enumerate(objects):
        similarity = pred[indices].mean(axis=0)
        sim_scores[idx] = similarity

    return sim_scores


def extract_clip_feature(clip_model, labelset: str | list[str]):
    # "ViT-L/14@336px" # the big model that OpenSeg uses

    if isinstance(labelset, str):
        lines = labelset.split(",")
    elif isinstance(labelset, list):
        lines = labelset
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


@click.command()
@click.option("--embeddings", help="path to embeddings folder")
@click.option("--metadata", help="path to metadata file")
@click.option("--scene-dir", "scene_dir", help="path to scene directory")
@click.option("--mask3d-dir", "mask3d_dir", help="path to mask3d predictions")
@click.option("--clip-model", default="ViT-B/16", help="CLIP model name")
@click.option("--split", default="test", help="dataset split")
@click.option("--box-source", "box_source", default="scannet", help="dataset source for boxes")
@click.option("--output", help="path to output folder")
def generate_predictions(embeddings, metadata, scene_dir, mask3d_dir, clip_model, split, box_source, output):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load CLIP model
    print("Loading CLIP {} model...".format(clip_model))
    clip_pretrained, _ = clip.load(clip_model, device=device, jit=False)

    # load metadata
    scenes = SceneCollection("ovfgvg", metadata, splits=[split])

    # iterate through scenes, compute similarity, and aggregate per box
    output_container = []
    for scene_id in tqdm(scenes.scene_ids[split]):
        # load embeddings
        embeddings_path = os.path.join(embeddings, f"{scene_id}.h5")
        hdf5_file = h5py.File(embeddings_path, "r")
        points = hdf5_file["points"]["points"][:]

        if "clip" not in hdf5_file["clip"]:
            breakpoint()
        clip_embeddings = hdf5_file["clip"]["clip"][:]
        point_embeddings = torch.from_numpy(clip_embeddings).to(device)  # (num_points, 512)

        # load description and boxes
        descriptions = scenes.get_annotations_by_scene(split, scene_id)
        dataset_utils = get_dataset(box_source, scene_dir=scene_dir, mask3d_directory=mask3d_dir)
        all_boxes = dataset_utils.load_object_boxes(scene_id)
        boxes = []
        for box in all_boxes.values():
            vertex_mask = box.contains(points)
            box_indices = np.where(vertex_mask)[0]
            boxes.append((box, box_indices))

        # compute text features
        text_features = extract_clip_feature(
            clip_pretrained, [desc.text for desc in descriptions]
        )  # (num_prompts, 512)

        # compute similarity
        similarities = point_embeddings.half() @ text_features.t()
        similarities = similarities.detach().cpu().numpy()
        box_scores = assign_clusters(boxes, similarities)
        selected_boxes = np.argmax(box_scores, axis=0)  # TODO: need to change this to support zero or multiple boxes

        for idx in range(len(descriptions)):
            predicted_box = boxes[selected_boxes[idx]][0]
            output_container.append(
                {
                    "prompt_id": descriptions[idx].id,
                    "scene_id": descriptions[idx].scene_id,
                    "prompt": descriptions[idx].text,
                    "predicted_boxes": [predicted_box.center.tolist(), predicted_box.dimensions.tolist()],
                }
            )

        hdf5_file.close()

    with open(output, "w") as f:
        json.dump(output_container, f, indent=4)


if __name__ == "__main__":
    generate_predictions()
