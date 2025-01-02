import json
import os

import click
import numpy as np
import pandas as pd

from ovfgvg.data.utils import SensorData


@click.command()
@click.option("--scene-id", help="scene_id")
@click.option("--sens", help="path to sens file")
@click.option("--width", default=320, help="image width")
@click.option("--height", default=240, help="image height")
@click.option("--frame-skip", default=20, help="number of frames to skip")
@click.option("--output", help="path to output folder")
def process_scene_main(scene_id, sens, width, height, frame_skip, output):
    return process_scene(scene_id, sens, width, height, frame_skip, output)


def process_scene(scene_id, sens, width, height, frame_skip, output):
    output_color_path = os.path.join(output, scene_id, "color")
    os.makedirs(output_color_path, exist_ok=True)
    output_depth_path = os.path.join(output, scene_id, "depth")
    os.makedirs(output_depth_path, exist_ok=True)
    output_pose_path = os.path.join(output, scene_id, "pose")
    os.makedirs(output_pose_path, exist_ok=True)
    intrinsic_dir = os.path.join(output, scene_id, "intrinsic")
    os.makedirs(intrinsic_dir, exist_ok=True)

    size = [width, height]

    sensor_data = SensorData(sens)
    _, intrinsic_color = sensor_data.export_color_images(output_color_path, image_size=size, frame_skip=frame_skip)
    _, intrinsic_depth = sensor_data.export_depth_images(output_depth_path, image_size=size, frame_skip=frame_skip)
    sensor_data.export_poses(output_pose_path, frame_skip=frame_skip)

    # save intrinsic matrix
    intrinsic_color_path = os.path.join(intrinsic_dir, "intrinsic_color.txt")
    intrinsic_depth_path = os.path.join(intrinsic_dir, "intrinsic_depth.txt")
    np.savetxt(intrinsic_color_path, intrinsic_color)
    np.savetxt(intrinsic_depth_path, intrinsic_depth)


if __name__ == "__main__":
    process_scene_main()
