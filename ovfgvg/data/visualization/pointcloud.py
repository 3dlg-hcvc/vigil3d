import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch


def export_pointcloud(name, points, colors=None, normals=None):
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)


def visualize_labels(
    u_index: list[int], labels: list[str], palette: list[int], out_name: str, loc: str = "lower left", ncol: int = 7
):
    """Generate an image of the legend mapping labels to colors for the point cloud.

    :param u_index: unique label indices in point cloud
    :param labels: list mapping index to label names
    :param palette: list of colors associated with each label. The list is flattened, so the index
    i maps to pixel colors [3*i:3*i+3].
    :param out_name: path at which to save legends
    :param loc: location for legend in matplotlib figure, defaults to "lower left"
    :param ncol: number of columns in legend, defaults to 7
    """
    patches = []
    for i, index in enumerate(u_index):
        label = labels[index]
        cur_color = [palette[index * 3] / 255.0, palette[index * 3 + 1] / 255.0, palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    plt.figure()
    plt.axis("off")
    legend = plt.legend(
        frameon=False, handles=patches, loc=loc, ncol=ncol, bbox_to_anchor=(0, -0.3), prop={"size": 5}, handlelength=0.7
    )
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, 5, 5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(out_name, bbox_inches=bbox, dpi=300)
    plt.close()


def convert_labels_with_palette(input, palette):
    """Get image color palette for visualizing masks"""

    new_3d = np.zeros((input.shape[0], 3))
    u_index = np.unique(input)
    for index in u_index:
        if index == 255:
            index_ = 20
        else:
            index_ = index

        new_3d[input == index] = np.array(
            [palette[index_ * 3] / 255.0, palette[index_ * 3 + 1] / 255.0, palette[index_ * 3 + 2] / 255.0]
        )

    return new_3d
