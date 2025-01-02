from .base import DatasetUtils
from .box_adapters import ClusteringMixin, Mask3dMixin
from .scannet import SensorData, ScanNet
from .scannetpp import ScanNetPP


# TODO: should be able to create these all dynamically
class ScanNetClustering(ClusteringMixin, ScanNet):
    pass


class ScanNetMask3D(Mask3dMixin, ScanNet):
    pass


class ScanNetPPClustering(ClusteringMixin, ScanNetPP):
    pass


class ScanNetPPMask3D(Mask3dMixin, ScanNetPP):
    pass


def get_dataset(name: str, **kwargs) -> DatasetUtils:
    # FIXME: this is a bad way of doing this; would be better to have some kind of string parsing
    if name == "scannet":
        return ScanNet(**kwargs)
    elif name == "scannetpp":
        return ScanNetPP(**kwargs)
    elif name == "scannet_clustering":
        return ScanNetClustering(**kwargs)
    elif name == "scannetpp_clustering":
        return ScanNetPPClustering(**kwargs)
    elif name == "scannet_mask3d":
        return ScanNetMask3D(**kwargs)
    elif name == "scannetpp_mask3d":
        return ScanNetPPMask3D(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


__all__ = ["DatasetUtils", "SensorData", "ScanNet", "ScanNetPP"]
