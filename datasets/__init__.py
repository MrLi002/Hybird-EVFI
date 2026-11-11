from .common import RawVFIData, shared_time_collate


def get_dataset(args, training=True, root=None):
    skips = args.skips if getattr(args,'skips',None) else ([9,11,15,23,31] if args.dataset=='gopro' else [1,3])
    return RawVFIData(root or args.data_root, dataset=args.dataset, skips=skips,
        voxel_bins=args.voxel_bins, num_targets=args.num_targets if training else None,
        crop_size=args.crop_size if training else None, training=training,
        augmentation=not getattr(args,'no_augmentation',False),
        legacy_event_time=args.legacy_event_time, frame_timestamps=args.frame_timestamps,
        image_fps=args.image_fps, time_offset=args.time_offset,
        frame_time_scale=args.frame_time_scale, event_time_scale=args.event_time_scale,
        event_coords=args.event_coords, polarity=args.polarity,
        scene_file=getattr(args,'scene_file',None), eval_stride=getattr(args,'eval_stride',None),
        gopro_odd_targets=getattr(args,'gopro_odd_targets',False),
        exclude_bad_frames=not getattr(args,'include_bad_frames',False))

