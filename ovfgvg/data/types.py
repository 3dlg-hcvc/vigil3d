import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Generator, Optional, Self

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from ovfgvg.data.visualization.bbox import save_bounding_box
from ovfgvg.utils import rescale, encode_mask, decode_mask


@dataclass
class OrientedBBox:
    """
    Dataclass for oriented bounding box.

    Parametrization is based on the center of the box, the half-dimensions along each axis, and the rotation of the
    box. If a rotation is not specified, the object is equivalent to an axis-aligned bounding box (AABB).
    """

    center: np.ndarray  # [3, ]
    half_dims: np.ndarray  # [3, ]
    rotation: Rotation = field(default_factory=lambda: Rotation.from_matrix(np.eye(3)))

    @property
    def dimensions(self):
        return 2 * self.half_dims

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Returns mask over points which are contained within the bounding box."""

        assert points.shape[1] == 3

        centered_pts = points - self.center
        oriented_pts = self.rotation.apply(centered_pts, inverse=True)
        return np.all(np.abs(oriented_pts) < self.half_dims, axis=1)

    @classmethod
    def from_dict(cls, data):
        return cls(
            center=np.array(data["center"]),
            half_dims=np.array(data["half_dims"]),
            rotation=Rotation.from_quat(np.array(data["rotation"])),
        )

    def to_dict(self):
        return {
            "center": self.center.tolist(),
            "half_dims": self.half_dims.tolist(),
            "rotation": self.rotation.as_quat().tolist(),
        }

    def to_array(self):
        return np.concatenate([self.center, self.half_dims, self.rotation.as_quat()])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Self:
        return cls(
            center=arr[:3],
            half_dims=arr[3:6],
            rotation=Rotation.from_quat(arr[6:]),
        )

    def to_ply(self, output: str):
        save_bounding_box(output, self.center, self.half_dims, self.rotation.as_matrix())

    @classmethod
    def from_mask(cls, coords: np.ndarray) -> Self:
        xmin = np.min(coords[:, 0])
        ymin = np.min(coords[:, 1])
        zmin = np.min(coords[:, 2])
        xmax = np.max(coords[:, 0])
        ymax = np.max(coords[:, 1])
        zmax = np.max(coords[:, 2])

        centroid = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
        dimensions = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

        return OrientedBBox(center=centroid, half_dims=dimensions / 2)

    @classmethod
    def cluster_boxes(
        cls, coords: np.ndarray, epsilon: float = 0.05, min_samples: int = 15
    ) -> tuple[list[Self], np.ndarray]:
        """Extracts oriented bounding boxes from point cloud based on mask."""
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        clusters = dbscan.fit(coords)
        labels = clusters.labels_

        # Initialize empty lists to store centroids and extends of each cluster
        boxes = []

        for cluster_id in set(labels):
            if cluster_id == -1:  # Ignore noise
                continue

            members = coords[labels == cluster_id]
            boxes.append(cls.from_mask(members))

        return boxes, labels


@dataclass
class LabeledOrientedBBox(OrientedBBox):
    id: str = None
    label: str | int = None

    def __post_init__(self):
        if self.id is None or self.label is None:
            raise ValueError(
                "Must specify an id or label. If neither is needed, consider using an OrientedBBox instead."
            )

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            label=data["label"],
            center=np.array(data["center"]),
            half_dims=np.array(data["half_dims"]),
            rotation=Rotation.from_quat(np.array(data["rotation"])),
        )

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            **super().to_dict(),
        }

    @classmethod
    def from_box(cls, id: str, label: str | int, box: OrientedBBox) -> Self:
        return cls(
            id=id,
            label=label,
            center=box.center,
            half_dims=box.half_dims,
            rotation=box.rotation,
        )

    def get_box(self) -> OrientedBBox:
        return OrientedBBox(center=self.center, half_dims=self.half_dims, rotation=self.rotation)

    @classmethod
    def from_mask(cls, id: str, label: str | int, coords: np.ndarray) -> Self:
        box = OrientedBBox.from_mask(coords)
        return cls.from_box(id, label, box)

    @classmethod
    def cluster_boxes(
        cls, coords: np.ndarray, epsilon: float = 0.05, min_samples: int = 15
    ) -> tuple[list[Self], np.ndarray]:
        """Extracts oriented bounding boxes from point cloud based on mask."""
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        clusters = dbscan.fit(coords)
        labels = clusters.labels_

        # Initialize empty lists to store centroids and extends of each cluster
        boxes = []

        for cluster_id in set(labels):
            if cluster_id == -1:  # Ignore noise
                continue

            members = coords[labels == cluster_id]
            boxes.append(cls.from_mask(id=cluster_id, label=None, coords=members))

        return boxes, labels


@dataclass
class Entity:
    is_target: bool
    # required
    boxes: list[OrientedBBox] = None
    mask: np.ndarray | None = None

    # optional
    target_name: str | None = (
        None  # referential name of target; required because entity could consist of multiple objects
    )
    labels: list[str] = None  # TODO: could just change this to LabeledOrientedBBox
    indexes: list[int] | None = None  # indices into text prompt by word
    ids: list[str | int] = None

    def __post_init__(self):
        if self.mask is None and self.boxes is None:
            raise ValueError(
                "Must specify at least one of mask or bbox. If no target object exists, then the mask should be a 0 "
                "mask, or bboxes should be an empty list."
            )

    @classmethod
    def from_dict(cls, data) -> Self:
        return cls(
            is_target=data.get("is_target"),
            target_name=data.get("target_name"),
            ids=data.get("ids"),
            labels=data.get("labels"),
            indexes=data.get("indexes"),
            boxes=[OrientedBBox.from_dict(box) for box in data["boxes"]] if data["boxes"] is not None else None,
            mask=decode_mask(data["mask"]) if data.get("mask") else None,
        )

    def to_dict(self):
        return {
            "is_target": self.is_target,
            "ids": self.ids,
            "target_name": self.target_name,
            "labels": self.labels,
            "indexes": self.indexes,
            "boxes": [box.to_dict() for box in self.boxes] if self.boxes is not None else None,
            "mask": encode_mask(self.mask) if self.mask is not None else None,
        }


@dataclass
class DescriptionPair:
    """
    Class for representing pairs of descriptions and objects.

    At a minimum, a description pair should include one of a mask or bbox and a text description. More generally, this
    class supports associating any number of objects with a single text description. This includes notably the
    following scenarios:
    * one target object (binary mask or box) and a grounding description
    * one target object, auxiliary objects, and a grounding description
    * multiple objects (no specific target), and a grounding description which is a list of classes
    * no target object and a grounding description

    The following scenarios could also be supported with minor modifications:
    * multiple target objects - need to track multiple boxes corresponding to the same class mapping, and may want an
      instance mask in addition to the existing semantic mask
    * distractor mask - need to add a mask field for distractors

    Note that masks are specified with respect to a particular scene.
    """

    text: str

    id: Optional[str] = None
    scene_id: Optional[str] = None

    # size: [num_points,]; 0 corresponds to unlabeled, and i corresponds to i-1 in class mapping
    # i==1 represents the target object
    mask: Optional[np.ndarray] = None
    entities: list[Entity] = None

    metadata: Optional[dict[str, Any]] = None

    TARGET_INDEX = 1

    @property
    def has_target(self) -> bool:
        return any(entity.is_target for entity in self.entities) if self.entities else False

    @property
    def target(self) -> Entity:
        targets = []
        for entity in self.entities:
            if entity.is_target:
                targets.append(entity)
        if len(targets) == 0:
            raise ValueError("The queried description pair has no target object.")
        elif len(targets) > 1:
            raise ValueError("The queried description pair has multiple target objects.")
        return targets[0]

    @classmethod
    def from_dict(cls, data) -> Self:
        return cls(
            id=data.get("id"),
            scene_id=data.get("scene_id"),
            text=data["text"],
            entities=[Entity.from_dict(entity) for entity in data["entities"]] if data["entities"] else None,
            metadata=data.get("metadata"),
            mask=decode_mask(data["mask"]) if data.get("mask") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scene_id": self.scene_id,
            "text": self.text,
            "entities": [entity.to_dict() for entity in self.entities] if self.entities else None,
            "metadata": self.metadata,
            "mask": encode_mask(self.mask) if self.mask else None,
        }


@dataclass
class ImageView:
    color: np.ndarray  # [h, w, c], where c=3 in RGB order
    depth: np.ndarray  # [h, w], in mm; 0.0 indicates no depth
    intrinsic_color: np.ndarray  # [4, 4]
    intrinsic_depth: np.ndarray  # [4, 4]
    pose: np.ndarray  # [4, 4]; camera pose

    @property
    def shape(self):
        return self.color.shape[:2]

    @property
    def permuted_image(self):
        return self.image

    @cached_property
    def rgbd(self):
        if self.color.shape[:2] != self.depth.shape[:2]:
            raise ValueError(
                f"Dimensions of RGB and D images do not match: {self.color.shape[:2]} != {self.depth.shape[:2]}"
            )

        rgbd = np.zeros((*self.color.shape[:2], 4))
        rgbd[:, :, :3] = self.color
        rgbd[:, :, 3] = self.depth / 1000.0  # convert to meters
        return rgbd

    @classmethod
    def from_dict(cls, data):
        return cls(
            color=data["image"],
            depth=data["depth"],
            intrinsic_color=data["intrinsic_color"],
            intrinsic_depth=data["intrinsic_depth"],
            pose=data["pose"],
        )

    def to_dict(self):
        return {
            "image": self.color,
            "depth": self.depth,
            "intrinsic_color": self.intrinsic_color,
            "intrinsic_depth": self.intrinsic_depth,
            "pose": self.pose,
        }


@dataclass
class Scene:
    """Generic scene representation to support many-to-one imports and one-to-many exports.

    The main rationale for this class is to reduce the implementation complexity to O(D+M), where D is the number of datasets and M is the number of models, rather than a many-to-many implementation which would be O(DM).

    The main elements of a scene are as follows:
    * points
        * coordinates (x,y,z)
        * color (r,g,b)
        * normals (nx,ny,nz)
    * descriptions
        * object mask
        * object bbox
            * center (x,y,z)
            * half dims (delta_x, delta_y, delta_z)
            * rotation
            * class
        * description
    * class_mapping

    Note that descriptions and masks/boxes+class_mapping are technically redundant, as any of the latter can be written
    as the former. We opt to keep both as the latter form can sometimes be more natural to provide as input.
    """

    scene_id: str

    coords: np.ndarray  # [num_points, 3]
    colors: Optional[np.ndarray] = None  # [num_points, 3]; RGB order, normalized to [-1, 1]
    _colors: np.ndarray = field(init=False, repr=False)
    normals: Optional[np.ndarray] = None

    boxes: list[LabeledOrientedBBox] = None

    views: Optional[list[ImageView]] = None

    @property
    def colors(self):
        if self._colors is not None:
            return self._colors

        return np.zeros_like(self.coords)

    @colors.setter
    def colors(self, colors: Optional[np.ndarray]):
        self._colors = colors

    @property
    def num_views(self):
        if not self.views:
            return 0
        return len(self.views)

    @property
    def image_size(self):
        if not self.views:
            raise ValueError(f"No views were provided in scene: {self.scene_id}")
        if any(self.views[0].shape != v.shape for v in self.views):
            raise ValueError(f"Some of the views have different shapes in scene: {self.scene_id}")

        return self.views[0].shape

    @staticmethod
    def mask_from_box(coords: np.ndarray, *boxes: OrientedBBox):
        annotations = np.zeros(
            (coords.shape[0],),
            dtype=int,
        )  # binary mask
        for box in boxes:
            assignment_mask = box.contains(coords)
            annotations[assignment_mask] = 1
        return annotations

    @staticmethod
    def box_from_mask(coords: np.ndarray, mask: np.ndarray):
        # assumes that there is only one object represented by the point cloud. In theory, we can use clustering to
        # identify multiple objects here
        obj_pc = coords[mask]
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])

        centroid = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
        dimensions = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

        return OrientedBBox(
            center=centroid,
            half_dims=dimensions / 2,
        )

    def colors_from_range(self, min_: int | float, max_: int | float, dtype=None):
        """Return colors within specified color range."""
        # self.colors starts as [-1, 1]
        colors = rescale(self.colors, -1, 1, min_, max_)

        if dtype:
            return colors.astype(dtype)
        else:
            return colors

    def filter(self, criteria: Callable):
        self.descriptions = [desc for desc in self.descriptions if criteria(desc)]
        return len(self.descriptions) > 0

    @classmethod
    def from_dict(cls, data):
        return cls(
            scene_id=data["scene_id"],
            coords=data["coords"],
            colors=data["color"],
            normals=data["normals"],
            boxes=data.get("boxes"),
            views=[ImageView.from_dict(view) for view in data["views"]],
        )

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "coords": self.coords,
            "color": self.colors,
            "normals": self.normals,
            "boxes": self.boxes,
            "views": [view.to_dict() for view in self.views] if self.views else [],
        }

    @classmethod
    def from_file(cls, inp_path: str):
        data = torch.load(inp_path)
        return cls.from_dict(data)

    def export(self, out_path: str):
        torch.save(self.to_dict(), out_path)


class SceneCollection:
    """
    Collection of scenes from a given dataset.

    A collection of scenes is specified through a directory, which includes one folder for each split. Each split
    folder should include a set of files, each of which corresponds to a single scene. Such files are parsed using the
    `id_parser` field passed to the constructor, which by default is of the form <scene_id>.pth. Additionally, the split
    folder could contain a metadata file (`metadata.json`) which specifies the annotation metadata for each scene.

    While each raw dataset has a different format, we rely on the preprocessing in this repository to convert each into
    the specified form here. We use separate files for each scene because the scenes have variable structure (e.g.
    different numbers of points, annotations, object classes, free-text fields, etc.) which make it difficult to
    utilize a more tabular structured format such as hdf5.

    An example scene directory could appear as follows:
    <dir>
    ├── train
    │   ├── scene1.pth
    │   ├── scene2.pth
    │   ├── ...
    │   └── metadata.json
    ├── val
    │   ├── scene3.pth
    │   ├── ...
    │   └── metadata.json

    Each scene file should be a dictionary directly importable into the Scene class (see `Scene.from_dict()` for more
    details), and each metadata.json file is specific to a split and should have the following form:
    {
        "num_annotations": number of raw annotations,
        "num_scenes": number of scenes,
        "scenes": {
            <scene_id>: {
                "annotations": [
                    "id": unique index of annotation within dataset split,
                    "description": full text description of annotation,
                    "object_id": target object_id,
                    "label": label name of target object,
                ]
            }
        },
        "annotation_lookup": [
            (<scene_id>, <annotation rel_id>)
        ]
    }

    The scene files should be comprehensive in terms of the annotation data stored, so here we primarily use the
    dataset-level metadata as we do not want to have to load every scene file in order to determine the number of
    annotations in it.
    """

    def __init__(
        self,
        name: str,  # provided for logging purposes only
        path: str,
        *,
        splits: Optional[list[str]] = None,
        id_parser: str = r"^(.*)\.pth$",
        metadata: Optional[str] = "metadata.json",
        filter_: Optional[Callable[[list[str], dict], tuple[list[str], dict]]] = None,
    ) -> None:
        self.name = name
        self.dir = path

        if splits is None:
            self.splits = [
                folder_name
                for folder_name in os.listdir(self.dir)
                if os.path.isdir(os.path.join(self.dir, folder_name))
            ]
        else:
            self.splits = splits

        self.scene_ids = {}
        self.scene_to_path = {}
        self.metadata = {}
        self.prompt_id_mapping = {}
        for split in self.splits:
            self.scene_ids[split] = []
            for file in os.listdir(os.path.join(self.dir, split)):
                res = re.search(id_parser, file)
                if res:
                    scene_id = res.group(1)
                    self.scene_ids[split].append(scene_id)
                    self.scene_to_path[scene_id] = os.path.join(self.dir, split, file)

            metadata_path = os.path.join(self.dir, split, metadata) if metadata else None
            if metadata and os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata[split] = json.load(f)

                # filter list of scenes and set of annotations
                # take intersection of scenes with point clouds and metadata UNLESS there are no point clouds, in which case just use metadata
                scene_ids_metadata = set([annot["scene_id"] for annot in self.metadata[split]["grounding"]])
                self.scene_ids[split] = (
                    list(scene_ids_metadata & set(self.scene_ids[split]))
                    if self.scene_ids[split]
                    else list(scene_ids_metadata)
                )
                if filter_ is not None:
                    self.apply_filter(filter_, splits=[split])
                else:
                    for i, annot in enumerate(self.metadata[split]["grounding"]):
                        if annot.get("id"):
                            self.prompt_id_mapping[annot["id"]] = i

                logging.info(f"Metadata file for dataset={self.name}, {split=} loaded: {metadata_path}")
            elif metadata is None:
                logging.warning(f"Metadata file for dataset={self.name}, {split=} not specified.")
                self.metadata[split] = None
            else:
                logging.warning(f"Metadata file for dataset={self.name}, {split=} not found: {metadata_path}")
                self.metadata[split] = None

    def apply_filter(self, filter_: Callable[[list[str], dict], tuple[list[str], dict]], splits: Optional[list[str]] = None):
        splits = splits if splits is not None else self.splits
        for split in splits:
            filtered_scene_ids, filtered_metadata = filter_(
                self.scene_ids[split], self.metadata[split]["grounding"]
            )
            self.scene_ids[split] = filtered_scene_ids
            self.metadata[split]["grounding"] = filtered_metadata

            for i, annot in enumerate(self.metadata[split]["grounding"]):
                if annot.get("id"):
                    self.prompt_id_mapping[annot["id"]] = i

    def iter_annotations(
        self, split: Optional[str] = None, count: Optional[int] = None, shuffle: bool = False
    ) -> Generator[DescriptionPair, Any, None]:
        num_annotations = (
            self.get_num_annotations(split) if count is None else min(count, self.get_num_annotations(split))
        )
        if not shuffle:
            indices = range(num_annotations)
        else:
            indices = np.random.choice(np.arange(self.get_num_annotations(split)), num_annotations, replace=False)

        for idx in indices:
            yield self.get_annotation(split, idx)

    def iter_scenes(
        self, split: Optional[str] = None, count: Optional[int] = None, shuffle: bool = False
    ) -> Generator[Scene, Any, None]:
        num_scenes = self.get_num_scenes(split) if count is None else min(count, self.get_num_scenes(split))
        if not shuffle:
            indices = range(num_scenes)
        else:
            indices = np.random.choice(np.arange(self.get_num_scenes(split)), num_scenes, replace=False)

        for idx in indices:
            yield self.get_scene(split, idx)

    def get_num_scenes(self, split: Optional[str] = None):
        splits = self.splits if split is None else [split]
        return sum(len(self.scene_ids[split]) for split in splits)

    def get_num_annotations(self, split: Optional[str] = None):
        splits = self.splits if split is None else [split]

        for split in splits:
            if self.metadata[split] is None:
                raise AttributeError(f"No metadata provided for scene collection for dataset={self.name}, {split=}.")

        return sum(len(self.metadata[split]["grounding"]) for split in splits)

    def __contains__(self, scene_id: str) -> bool:
        return scene_id in self.scene_to_path

    def get_scene(self, split: str, index: int) -> Scene:
        """
        Get scene corresponding to specified scene index.

        :param split: split from which to pull scene
        :param index: unique index of the scene within the dataset split
        :return: scene corresponding to scene index
        """
        scene_id = self.scene_ids[split][index]
        return self.get_scene_by_id(scene_id)

    def get_scene_by_id(self, scene_id: str) -> Scene:
        """
        Get scene corresponding to specified scene_id.

        :param split: split from which to pull scene
        :param index: scene_id within the dataset split
        :return: scene corresponding to scene_id
        """
        path = self.scene_to_path[scene_id]
        return Scene.from_file(path)

    def get_annotation(self, split: str, index: int) -> DescriptionPair:
        """
        Get full scene corresponding to specified annotation index.

        :param split: split from which to pull scene
        :param index: unique index of the annotation within the dataset split
        :raises AttributeError: metadata file not loaded
        :return: scene corresponding to the annotation
        """
        if self.metadata[split] is None:
            raise AttributeError(f"No metadata provided for scene collection for dataset={self.name}, {split=}.")

        return DescriptionPair.from_dict(self.metadata[split]["grounding"][index])

    def get_annotation_by_id(self, split: str, annot_id: str) -> DescriptionPair:
        return self.get_annotation(split, self.prompt_id_mapping[annot_id])

    def get_scene_and_annotation(self, split: str, annot_index: int) -> tuple[Scene, DescriptionPair]:
        """
        Get scene and annotation corresponding to specified annotation index.

        The list of descriptions in the scene should only have one entry corresponding to the specified annotation.

        :param split: split from which to pull scene
        :param index: unique index of the annotation within the dataset split
        :raises AttributeError: metadata file not loaded
        :return: scene corresponding to the annotation, including only the specified annotation.
        """
        if self.metadata[split] is None:
            raise AttributeError(f"No metadata provided for scene collection for dataset={self.name}, {split=}.")

        annotation = self.get_annotation(split, annot_index)
        scene = self.get_scene_by_id(annotation.scene_id)

        return scene, annotation

    def get_annotations_by_scene(self, split: str, scene_id: str) -> list[DescriptionPair]:
        """
        useful for getting all description pairs for a scene without requiring the point cloud data
        """
        return [
            DescriptionPair.from_dict(annot)
            for annot in self.metadata[split]["grounding"]
            if annot["scene_id"] == scene_id
        ]


class SceneCollections:
    def __init__(self, scene_collections: list[SceneCollection]) -> None:
        self.scene_collections = scene_collections

    def apply_filter(self, filter_: Callable[[list[str], dict], tuple[list[str], dict]], splits: Optional[list[str]] = None):
        for sc in self.scene_collections:
            sc.apply_filter(filter_, splits=splits)

    def get_last_scene_indices(self, split: str):
        return [sc.get_num_scenes(split) for sc in self.scene_collections]

    def get_last_annotation_indices(self, split: str):
        return [sc.get_num_annotations(split) for sc in self.scene_collections]

    def get_num_scenes(self, split: Optional[str] = None):
        return sum(sc.get_num_scenes(split) for sc in self.scene_collections)

    def get_num_annotations(self, split: Optional[str] = None):
        return sum(sc.get_num_annotations(split) for sc in self.scene_collections)

    def get_scene(self, split: str, index: int) -> Scene:
        boundaries = self.get_last_scene_indices(split)
        for i, last_index in enumerate(boundaries):
            if index < last_index:
                first_index = 0 if i == 0 else boundaries[i - 1]
                return self.scene_collections[i].get_scene(split, index - first_index)
        raise ValueError(f"Index {index} is out of bounds for scenes.")

    def get_annotation(self, split: str, index: int) -> DescriptionPair:
        """
        Get annotation corresponding to specified annotation index which is global across all scene
        collections.

        The list of descriptions in the scene should only have one entry corresponding to the specified annotation.

        :param split: split from which to pull scene
        :param index: unique index of the annotation within split but across all scene collections
        :param return_id: if True, return annotation index relative to scene
        :raises AttributeError: metadata file for relevant scene collection not loaded
        :return: scene corresponding to the annotation, including only the specified annotation.
        """
        boundaries = self.get_last_annotation_indices(split)
        for i, last_index in enumerate(boundaries):
            if index < last_index:
                first_index = 0 if i == 0 else boundaries[i - 1]
                return self.scene_collections[i].get_annotation(split, index - first_index)
        raise ValueError(f"Index {index} is out of bounds for annotations.")

    def get_scene_and_annotation(self, split: str, index: int) -> tuple[Scene, DescriptionPair]:
        """
        Get scene and annotation corresponding to specified annotation index which is global across all scene
        collections.

        The list of descriptions in the scene should only have one entry corresponding to the specified annotation.

        :param split: split from which to pull scene
        :param index: unique index of the annotation within split but across all scene collections
        :param return_id: if True, return annotation index relative to scene
        :raises AttributeError: metadata file for relevant scene collection not loaded
        :return: scene corresponding to the annotation, including only the specified annotation.
        """
        boundaries = self.get_last_annotation_indices(split)
        for i, last_index in enumerate(boundaries):
            if index < last_index:
                first_index = 0 if i == 0 else boundaries[i - 1]
                return self.scene_collections[i].get_scene_and_annotation(split, index - first_index)
        raise ValueError(f"Index {index} is out of bounds for annotations.")

    def get_annotations_by_scene(self, split: str, scene_id: str) -> list[DescriptionPair]:
        for sc in self.scene_collections:
            if scene_id in sc.scene_ids[split]:
                return sc.get_annotations_by_scene(split, scene_id)
