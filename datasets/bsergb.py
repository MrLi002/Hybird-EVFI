"""OF-AF BS-ERGB constructor, interval indexing, crop and packed sample format."""
from .common import FrameAlignedTrain, FrameAlignedTest


class BSERGB_Train(FrameAlignedTrain):
    def __init__(self, voxel_bins, data_root, scene, skip_frame, crop_size=256, augmentation=True):
        super().__init__(voxel_bins, data_root, scene, skip_frame, bsergb=True,
                         crop_size=crop_size, augmentation=augmentation)


class BSERGB_Test(FrameAlignedTest):
    def __init__(self, voxel_bins, data_root, scene, skip_frame):
        super().__init__(voxel_bins, data_root, scene, skip_frame, bsergb=True)
