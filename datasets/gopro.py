"""GoPro organized like OF-AF BS-ERGB: images/ plus frame-aligned events/."""
from .common import FrameAlignedTrain, FrameAlignedTest


class GOPRO_Train(FrameAlignedTrain):
    def __init__(self, voxel_bins, data_root, scene, skip_frame, crop_size=256, augmentation=True):
        super().__init__(voxel_bins, data_root, scene, skip_frame, crop_size=crop_size,
                         augmentation=augmentation)


class GOPRO_Test(FrameAlignedTest):
    def __init__(self, voxel_bins, data_root, scene, skip_frame):
        super().__init__(voxel_bins, data_root, scene, skip_frame)
