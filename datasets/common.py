"""OF-AF BS-ERGB-style frame-aligned loading for both datasets.

Scene layout: images/*.png and events/*.npz. Event j covers image j to j+1.
Per-scene loaders return (channel-packed RGB, interval voxel, target times).
RawVFIData adapts these tuples to the hybrid trainer's named dictionary.
"""
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from util.event import EventSequence
from util.voxelization import to_voxel_grid

DATA_PROTOCOL = 'ofaf_frame_aligned_v1'
BAD_FRAMES = {'basket_09': {31, 32, 33, 34}, 'may29_rooftop_handheld_02': {17, 70},
              'may29_rooftop_handheld_03': {306}, 'may29_rooftop_handheld_05': {21}}


@dataclass
class Scene:
    path: Path
    images: list
    event_files: list


class FrameAlignedTrain(Dataset):
    """Same constructor and sample layout as OF-AF's BSERGB_Train."""
    def __init__(self, voxel_bins, data_root, scene, skip_frame, *, bsergb=False,
                 crop_size=256, augmentation=True):
        if skip_frame < 1 or voxel_bins < 2 or crop_size < 1:
            raise ValueError('skip_frame/crop_size must be positive and voxel_bins >= 2')
        self.skip_frame = self.cur_skip = int(skip_frame)
        self.crop = [crop_size, crop_size]
        self.voxel_bins, self.bsergb = voxel_bins, bsergb
        self.augmentation = augmentation
        self.data_root = Path(data_root).expanduser().resolve()
        self.scene_root = self.data_root / scene
        self.img_folder, self.event_folder = self.scene_root / 'images', self.scene_root / 'events'
        self.img_names = sorted(p.name for p in self.img_folder.glob('*.png'))
        self.event_names = sorted(p.name for p in self.event_folder.glob('*.npz'))
        if not self.img_names:
            raise ValueError(f'{self.img_folder}: no .png images')
        expected = len(self.img_names)-1
        if len(self.event_names) != expected:
            raise ValueError(f'{self.event_folder}: expected {expected} frame-aligned .npz files '
                             f'for {len(self.img_names)} images, got {len(self.event_names)}; '
                             'event j must cover image j to j+1 (merge high-rate files first)')
        with Image.open(self.img_folder / self.img_names[0]) as image:
            self.w, self.h = image.size

    def __len__(self):
        return max(0, len(self.img_names)-self.skip_frame-1)

    def get_img(self, index):
        if not 0 <= index < len(self.img_names)-self.cur_skip-1:
            raise IndexError(index)
        ind = list(range(index, index+self.cur_skip+2))
        images = []
        for i in ind:
            with Image.open(self.img_folder / self.img_names[i]) as image:
                images.append(torch.from_numpy(np.array(image.convert('RGB'), copy=True)))
        if any(image.shape != images[0].shape for image in images):
            raise ValueError(f'{self.scene_root}: image sizes change inside the interval')
        return torch.cat(images, dim=2), ind

    def get_events(self, ind, size=None):
        paths = [self.event_folder / name for name in self.event_names[ind[0]:ind[-1]]]
        height, width = (self.h, self.w) if size is None else (size[2]-size[0], size[3]-size[1])
        return EventSequence.from_npz_files(paths, height, width, bsergb=self.bsergb, size=size)

    def random_crop(self, imgs):
        h, w = imgs.shape[:2]
        if h < self.crop[0] or w < self.crop[1]:
            raise ValueError('crop_size exceeds image size')
        hs, ws = random.randint(0, h-self.crop[0]), random.randint(0, w-self.crop[1])
        he, we = hs+self.crop[0], ws+self.crop[1]
        return imgs[hs:he, ws:we], [hs, ws, he, we]

    def read_interval(self, index, training=True):
        imgs, ind = self.get_img(index)
        if training:
            imgs, size = self.random_crop(imgs)
        else:
            size = None
        events = self.get_events(ind, size)
        h, w = imgs.shape[:2]
        voxel = (torch.zeros(self.voxel_bins, h, w) if events is None
                 else to_voxel_grid(events, self.voxel_bins))
        imgs = imgs.permute(2, 0, 1).float()/255
        times = (torch.tensor(ind[1:-1])-ind[0])/(self.cur_skip+1)
        if training and self.augmentation:
            for dim in (1, 2):
                if random.uniform(0, 1) < .5:
                    imgs, voxel = imgs.flip(dim), voxel.flip(dim)
        return imgs, voxel, times

    def __getitem__(self, index):
        return self.read_interval(index)


class FrameAlignedTest(FrameAlignedTrain):
    """Full-frame samples; stride and BS-ERGB bad-frame rules follow OF-AF."""
    def __init__(self, voxel_bins, data_root, scene, skip_frame, *, bsergb=False):
        super().__init__(voxel_bins, data_root, scene, skip_frame, bsergb=bsergb,
                         augmentation=False)
        bad = BAD_FRAMES.get(self.scene_root.name, set()) if bsergb else set()
        self.starts = [i for i in range(0, len(self.img_names)-skip_frame-1, skip_frame)
                       if not any(j in bad for j in range(i, i+skip_frame+2))]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        return self.read_interval(self.starts[index], training=False)

    def get_batch(self, index):
        return tuple(x.unsqueeze(0) for x in self.read_interval(index, training=False))


class RawVFIData(Dataset):
    """Enumerate (skip, scene, start) like OF-AF's ConcatDataset.

    All targets are used by default; num_targets is an optional training cap.
    """
    def __init__(self, root, dataset='gopro', skips=None, voxel_bins=128,
                 num_targets=None, crop_size=256, training=True, augmentation=True):
        from .bsergb import BSERGB_Train, BSERGB_Test
        from .gopro import GOPRO_Train, GOPRO_Test
        if dataset not in ('gopro', 'bsergb'):
            raise ValueError('dataset must be gopro or bsergb')
        defaults = (31,) if training and dataset == 'gopro' else ((7, 15) if dataset == 'gopro' else (1, 3))
        self.skips = tuple(sorted(set(defaults if skips is None else skips)))
        if not self.skips or min(self.skips) < 1:
            raise ValueError('skips must contain positive integers')
        if num_targets is not None and num_targets < 1:
            raise ValueError('num_targets must be positive or omitted')
        if not training and dataset == 'gopro' and min(self.skips) < 2:
            raise ValueError('OF-AF GoPro evaluation selects target indices 1,3,...; use skips >= 2')
        self.root = Path(root).expanduser().resolve()
        self.dataset, self.training = dataset, training
        self.voxel_bins, self.num_targets = voxel_bins, num_targets
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        folders = ([self.root] if (self.root/'images').is_dir() else
                   [p for p in sorted(self.root.iterdir()) if (p/'images').is_dir()])
        if not folders:
            raise ValueError(f'{self.root}: no scenes with images/*.png and events/*.npz; pass a split root')
        cls = ((BSERGB_Train if training else BSERGB_Test) if dataset == 'bsergb'
               else (GOPRO_Train if training else GOPRO_Test))
        self.scenes, self.datasets, self.samples = [], [], []
        for folder in folders:
            probe = cls(voxel_bins, folder.parent, folder.name, self.skips[0],
                        **(dict(crop_size=crop_size, augmentation=augmentation) if training else {}))
            self.scenes.append(Scene(folder, [probe.img_folder/n for n in probe.img_names],
                                     [probe.event_folder/n for n in probe.event_names]))
        for skip in self.skips:
            for scene in self.scenes:
                child = cls(voxel_bins, scene.path.parent, scene.path.name, skip,
                            **(dict(crop_size=crop_size, augmentation=augmentation) if training else {}))
                child_index = len(self.datasets)
                self.datasets.append(child)
                self.samples.extend((child_index, i) for i in range(len(child)))
        if not self.samples:
            raise ValueError('no valid intervals; check split root, skips and frame count')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        child_index, local_index = self.samples[item]
        child = self.datasets[child_index]
        start = local_index if self.training else child.starts[local_index]
        imgs, voxel, times = child[local_index]
        frames = imgs.reshape(-1, 3, *imgs.shape[-2:])
        selected = list(range(child.cur_skip))
        if self.training and self.num_targets is not None:
            selected = sorted(random.sample(selected, min(self.num_targets, len(selected))))
        elif not self.training and self.dataset == 'gopro':
            selected = selected[1::2]
        return dict(im0=frames[0], im1=frames[-1], targets=frames[1:-1][selected],
                    event=voxel, times=times[selected], scene=child.scene_root.name,
                    start=start, skip=child.cur_skip,
                    target_names=[child.img_names[start+i+1] for i in selected])

    def manifest(self):
        return dict(protocol=DATA_PROTOCOL, dataset=self.dataset, root=str(self.root),
                    training=self.training, skips=self.skips, voxel_bins=self.voxel_bins,
                    samples=len(self), num_targets=self.num_targets if self.training else None,
                    event_time_normalization='min/max events remaining after spatial crop',
                    event_file_mapping='event j covers image j to image j+1',
                    eval_stride=None if self.training else 'skip',
                    eval_targets='zero-based odd intermediate indices' if not self.training and self.dataset == 'gopro' else 'all (unless explicit num_targets)',
                    scenes=[dict(name=s.path.name, root=str(s.path), frames=len(s.images),
                                 event_files=len(s.event_files)) for s in self.scenes])


def shared_time_collate(samples):
    from torch.utils.data import default_collate
    if any(s['times'].shape != samples[0]['times'].shape or not torch.allclose(s['times'], samples[0]['times'])
           for s in samples[1:]):
        raise ValueError('batch samples need identical target times; use batch_size=1 for mixed skips/target sampling')
    return default_collate(samples)
