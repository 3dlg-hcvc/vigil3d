import open3d as o3d
import numpy as np


def write_ply(verts, colors, indices, output_file):
    """
    Implementation taken from (ScanRefer)[https://github.com/daveredrum/ScanRefer].
    """
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, "w")
    file.write("ply \n")
    file.write("format ascii 1.0\n")
    file.write("element vertex {:d}\n".format(len(verts)))
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property uchar red\n")
    file.write("property uchar green\n")
    file.write("property uchar blue\n")
    file.write("element face {:d}\n".format(len(indices)))
    file.write("property list uchar uint vertex_indices\n")
    file.write("end_header\n")
    for vert, color in zip(verts, colors):
        file.write(
            "{:f} {:f} {:f} {:d} {:d} {:d}\n".format(
                vert[0], vert[1], vert[2], int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
        )
    for ind in indices:
        file.write("3 {:d} {:d} {:d}\n".format(ind[0], ind[1], ind[2]))
    file.close()


def get_bounding_box_mesh(centroid: list[float], extent: list[float], color: list[int] = None, radius: float = 0.015):
    """
    Implementation taken from (ScanRefer)[https://github.com/daveredrum/ScanRefer].

    output: str (path to save the ply file)
    centroid: [x, y, z]
    extent: [x, y, z]
    color: [r, g, b] (0-255)
    radius: float (thickness of bounding box edges)
    """

    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0, 0] = 1 + t * (x * x - 1)
            rot[0, 1] = z * s + t * x * y
            rot[0, 2] = -y * s + t * x * z
            rot[1, 0] = -z * s + t * x * y
            rot[1, 1] = 1 + t * (y * y - 1)
            rot[1, 2] = x * s + t * y * z
            rot[2, 0] = y * s + t * x * z
            rot[2, 1] = -x * s + t * y * z
            rot[2, 2] = 1 + t * (z * z - 1)
            return rot

        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks + 1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius * math.cos(theta), radius * math.sin(theta), height * i / stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
                indices.append(
                    np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32)
                )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if math.fabs(dotx) != 1.0:
                    axis = np.array([1, 0, 0]) - dotx * va
                else:
                    axis = np.array([0, 1, 0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]]),
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),
            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),
            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7]),
        ]
        return edges

    def get_bbox_corners(centroid, extent):
        xmin, xmax = centroid[0] - extent[0] / 2, centroid[0] + extent[0] / 2
        ymin, ymax = centroid[1] - extent[1] / 2, centroid[1] + extent[1] / 2
        zmin, zmax = centroid[2] - extent[2] / 2, centroid[2] + extent[2] / 2
        corners = []
        corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
        corners = np.concatenate(corners, axis=0)  # 8 x 3

        return corners

    if color is None:
        color = [0, 0, 255]
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(centroid, extent)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    return verts, colors, indices


def save_bounding_box(
    output: str, centroid: list[float], extent: list[float], color: list[int] = None, radius: float = 0.03
):
    """
    Implementation taken from (ScanRefer)[https://github.com/daveredrum/ScanRefer].

    output: str (path to save the ply file)
    centroid: [x, y, z]
    extent: [x, y, z]
    color: [r, g, b] (0-255)
    radius: float (thickness of bounding box edges)
    """

    verts, colors, indices = get_bounding_box_mesh(centroid, extent, color, radius)
    write_ply(verts, colors, indices, output)


def save_bounding_box_scannetpp(
    output: str, centroid: list[float], extent: list[float], color: list[int] = None, radius: float = 0.03
):
    extent = np.asarray([extent[0], extent[2], extent[1]])
    return save_bounding_box(output, centroid, extent, color, radius)
