import datetime
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import click
import h5py
import torch
from nerfstudio.utils.eval_utils import eval_setup
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ovfgvg.scripts.lerf.preprocess_scannet import process_scene
from ovfgvg.utils import Timer, estimate_num_parameters

CONSOLE = Console(width=120)


def export_point_cloud(
    config_path: str,  # path to YAML file
    output_path: str,  # path to output file
    num_points: int = 1000000,
    num_rays_per_batch: int = 32768,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    clip_embeddings_name: str = "clip",
    normal_output_name: Optional[str] = None,
):
    """Generate a point cloud from a NERF.

    This implementation is based on the ns-export function from nerfstudio and the modified point cloud
    export from https://github.com/sled-group/nerfstudio_export. We adapt it to be callable directly within Python and
    to focus purely on the data extraction for evaluation.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # setup pipeline
    print(f"Config: {config_path}")
    _, pipeline, _, _ = eval_setup(config_path)
    print(f"# parmeters: {estimate_num_parameters(pipeline.model)}")

    # Increase the batchsize to speed up the evaluation.
    pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = num_rays_per_batch

    # pylint: disable=too-many-statements
    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    points = []
    rgbs = []
    normals = []
    clips = []

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            # get point samples
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                outputs = pipeline.model(ray_bundle)

            # verify valid outputs
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"Could not find {rgb_output_name} in the model outputs",
                    justify="center",
                )
                CONSOLE.print(
                    f"Please set --rgb_output_name to one of: {outputs.keys()}",
                    justify="center",
                )
                return
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(
                    f"Could not find {depth_output_name} in the model outputs",
                    justify="center",
                )
                CONSOLE.print(
                    f"Please set --depth_output_name to one of: {outputs.keys()}",
                    justify="center",
                )
                return
            rgb = outputs[rgb_output_name]
            depth = outputs[depth_output_name]
            clip = outputs[clip_embeddings_name]

            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(
                        f"Could not find {normal_output_name} in the model outputs",
                        justify="center",
                    )
                    CONSOLE.print(
                        f"Please set --normal_output_name to one of: {outputs.keys()}",
                        justify="center",
                    )
                    return
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0
            point = ray_bundle.origins + ray_bundle.directions * depth

            # save into containers
            points.append(point)
            rgbs.append(rgb)
            clips.append(clip)
            if normal_output_name is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])

    # aggregate
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    clips = torch.cat(clips, dim=0)

    # save intermediate result to h5 file
    CONSOLE.print("Saving H5 file...")
    hdf5_file = h5py.File(output_path, "w")

    # FIXME: we don't need the group part at all
    points_group = hdf5_file.create_group("points")
    clip_group = hdf5_file.create_group("clip")  # embeddings
    rgb_group = hdf5_file.create_group("rgb")  # rendering

    points_group.create_dataset("points", data=points.detach().cpu().numpy())
    rgb_group.create_dataset("rgb", data=rgbs.detach().cpu().numpy())
    clip_group.create_dataset("clip", data=clips.detach().cpu().numpy())

    hdf5_file.close()
    CONSOLE.print("Done saving H5 file...")


@click.command()
@click.option("--metadata", help="path to metadata file")
@click.option("--data", help="path to data folder")
@click.option("--width", default=320, help="image width")
@click.option("--height", default=240, help="image height")
@click.option("--frame-skip", default=20, help="number of frames to skip")
@click.option("--num-points", default=1000000, help="number of points to sample to generate embeddings")
@click.option("--num-rays-per-batch", default=32768, help="number of rays to sample per batch during 'inference'")
@click.option("--data-dir", help="path to output folder")
@click.option("--checkpoint-dir", help="path to checkpoint directory")
@click.option("--cache-dir", help="path to prediction cache directory")
@click.option("--run-all", is_flag=True, help="if True, rerun all processes even if files already exist")
def run_lerf(
    metadata,
    data,
    width,
    height,
    frame_skip,
    num_points,
    num_rays_per_batch,
    data_dir,
    checkpoint_dir,
    cache_dir,
    run_all,
):
    timer = Timer()
    with open(metadata, "r") as f:
        metadata_dict = json.load(f)

    scene_ids = set([prompt["scene_id"] for prompt in metadata_dict["grounding"]])

    for scene_id in scene_ids:
        if run_all or any(
            not os.path.exists(os.path.join(data_dir, scene_id, dir_))
            for dir_ in ["color", "depth", "pose", "intrinsic"]
        ):
            # FIXME: specific to ScanNet data
            process_scene(
                scene_id, os.path.join(data, scene_id, f"{scene_id}.sens"), width, height, frame_skip, data_dir
            )
        timer.start()

        if run_all or not os.path.exists(os.path.join(checkpoint_dir, scene_id)):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            subprocess.run(
                [
                    "ns-train",
                    "lerf",
                    "--output-dir",
                    checkpoint_dir,
                    "--experiment-name",
                    scene_id,
                    "--timestamp",
                    timestamp,
                    "--viewer.quit-on-train-completion",
                    "True",
                    "scannet-data",
                    "--data",
                    os.path.join(data_dir, scene_id),
                ]
            )
        else:
            # pull the latest trained version of the model
            timestamp = sorted(os.listdir(os.path.join(checkpoint_dir, scene_id, "lerf")))[-1]

        # export point cloud and embeddings from trained model
        os.makedirs(cache_dir, exist_ok=True)
        embeddings_file = os.path.join(cache_dir, f"{scene_id}.h5")
        if True or run_all or not os.path.exists(embeddings_file):
            config_path = Path(checkpoint_dir) / scene_id / "lerf" / timestamp / "config.yml"
            export_point_cloud(
                config_path, embeddings_file, num_points=num_points, num_rays_per_batch=num_rays_per_batch
            )

        timer.end()

    print(timer.print())


if __name__ == "__main__":
    run_lerf()
