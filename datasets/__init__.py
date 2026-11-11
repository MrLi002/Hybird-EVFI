from .common import DATA_PROTOCOL, RawVFIData, shared_time_collate
from .bsergb import BSERGB_Train, BSERGB_Test
from .gopro import GOPRO_Train, GOPRO_Test


def get_dataset(args, training=True, root=None):
    return RawVFIData(root or args.data_root, dataset=args.dataset,
        skips=getattr(args, 'skips', None), voxel_bins=args.voxel_bins,
        num_targets=getattr(args, 'num_targets', None) if training else None,
        crop_size=args.crop_size if training else None, training=training,
        augmentation=not getattr(args, 'no_augmentation', False))
