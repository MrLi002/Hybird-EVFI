"""Raw RGB/event datasets with explicit physical-time voxelization.

Scene layout: images/ (BS-ERGB) or imgs/ (GoPro), and events/ or
events_clean_8x_v2/. No train/validation/test directories are silently merged.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import random
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from util.event import load_events
from util.voxelization import events_to_voxel_grid

BAD_FRAMES = {'basket_09': {31,32,33,34}, 'may29_rooftop_handheld_02': {17,70},
              'may29_rooftop_handheld_03': {306}, 'may29_rooftop_handheld_05': {21}}


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(path))]


def read_timestamps(path):
    result = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.replace(',', ' ').split()
        value = None
        for candidate in (parts[-1], parts[0]):
            try:
                value = float(candidate)
                break
            except ValueError:
                pass
        if value is None:
            # Allow one explicit header, not corrupt records in the middle of the file.
            if not result:
                continue
            raise ValueError(f'{path}: invalid timestamp row {raw!r}')
        result.append(value)
    values = np.asarray(result, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any(np.diff(values)<=0):
        raise ValueError(f'{path}: timestamps must be finite and strictly increasing')
    return values


@dataclass
class Scene:
    path: Path
    images: list
    event_files: list
    frame_times: np.ndarray
    event_starts: object
    timestamp_source: str


class RawVFIData(Dataset):
    def __init__(self, root, dataset='gopro', skips=(9,11,15,23,31), voxel_bins=128,
                 num_targets=4, crop_size=256, training=True, augmentation=True,
                 legacy_event_time=False, frame_timestamps=None, image_fps=None,
                 time_offset=0., frame_time_scale=1., event_time_scale=1.,
                 event_coords='ofaf', polarity='auto', scene_file=None,
                 eval_stride=None, gopro_odd_targets=False, exclude_bad_frames=True):
        self.root = Path(root).expanduser().resolve()
        self.dataset, self.skips = dataset, tuple(sorted(set(int(x) for x in skips)))
        if dataset not in ('gopro', 'bsergb') or not self.skips or min(self.skips)<1:
            raise ValueError('dataset must be gopro/bsergb and skips must be positive')
        if num_targets is not None and num_targets<1:
            raise ValueError('num_targets must be positive or None')
        self.voxel_bins, self.num_targets = voxel_bins, num_targets
        self.training, self.augmentation, self.crop_size = training, augmentation and training, crop_size
        self.legacy_event_time = legacy_event_time
        self.event_time_scale, self.event_coords, self.polarity = event_time_scale, event_coords, polarity
        self.gopro_odd_targets = gopro_odd_targets and not training
        self.eval_stride = eval_stride
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        image_dir_name = 'imgs' if dataset == 'gopro' else 'images'
        folders = [self.root] if (self.root/image_dir_name).is_dir() else [
            p for p in sorted(self.root.iterdir(), key=natural_key) if (p/image_dir_name).is_dir()]
        if scene_file:
            requested = {line.strip() for line in Path(scene_file).read_text().splitlines()
                         if line.strip() and not line.startswith('#')}
            missing = requested-{p.name for p in folders}
            if missing:
                raise ValueError(f'scenes not under the specified root: {sorted(missing)}')
            folders = [p for p in folders if p.name in requested]
        if not folders:
            raise ValueError(f'no scenes found in {self.root}; pass the split root, e.g. GOPRO/train or BSERGB/1_TRAIN')
        self.scenes, self.samples = [], []
        for folder in folders:
            images = sorted([p for p in (folder/image_dir_name).iterdir()
                             if p.suffix.lower() in ('.png','.jpg','.jpeg')], key=natural_key)
            event_folder = folder/('events_clean_8x_v2' if dataset=='gopro' else 'events')
            events = sorted(event_folder.glob('*.npz'), key=natural_key)
            if len(images)<min(self.skips)+2:
                continue
            if not events:
                raise ValueError(f'{event_folder}: no .npz events')
            times, source = None, None
            candidates = [folder/frame_timestamps] if frame_timestamps else [
                folder/'images_timestamps.txt', folder/'timestamps_images.txt',
                folder/image_dir_name/'timestamps.txt', folder/'timestamps.txt']
            if frame_timestamps and not candidates[0].is_file():
                raise FileNotFoundError(candidates[0])
            for candidate in candidates:
                if candidate.is_file():
                    values = read_timestamps(candidate)*frame_time_scale
                    if len(values)==len(images):
                        times, source = values, str(candidate)
                        break
                    if frame_timestamps:
                        raise ValueError(f'{candidate}: expected {len(images)} frame timestamps, got {len(values)}')
            fps = image_fps if image_fps is not None else (240. if dataset=='gopro' else None)
            if times is None and fps is not None:
                if fps<=0:
                    raise ValueError('image_fps must be positive')
                times = np.arange(len(images), dtype=np.float64)/fps+time_offset
                source = f'explicit/known fps={fps}, time_offset={time_offset}'
            if times is None:
                if not legacy_event_time:
                    raise ValueError(f'{folder}: frame timestamps missing. Provide --frame_timestamps, or known --image_fps/--time_offset, or explicitly use --legacy_event_time.')
                times, source = np.arange(len(images), dtype=np.float64), 'legacy index times (not physical frame timestamps)'
            starts = None
            event_clock = folder/'timestamps_clean_8x_v2.txt'
            if dataset=='gopro' and event_clock.is_file():
                values = read_timestamps(event_clock)*event_time_scale
                if len(values) not in (len(events), len(events)+1):
                    raise ValueError(f'{event_clock}: expected one time per event file or boundary')
                starts = values[:len(events)]
            elif len(events)<len(images)-1:
                raise ValueError(f'{folder}: insufficient frame-aligned event files')
            elif dataset=='gopro' and len(events)>len(images):
                raise ValueError(f'{folder}: high-rate event files require timestamps_clean_8x_v2.txt')
            scene_index = len(self.scenes)
            self.scenes.append(Scene(folder, images, events, times, starts, source))
            if training:
                self.samples.extend((scene_index, index, None) for index in range(len(images)-min(self.skips)-1))
            else:
                for skip in self.skips:
                    stride = eval_stride if eval_stride is not None else skip
                    for index in range(0, len(images)-skip-1, stride):
                        bad = BAD_FRAMES.get(folder.name, set()) if exclude_bad_frames and dataset=='bsergb' else set()
                        if any(j in bad for j in range(index,index+skip+2)):
                            continue
                        self.samples.append((scene_index,index,skip))
        if not self.samples:
            raise ValueError('no valid intervals; check skips, split root and frame count')

    def __len__(self):
        return len(self.samples)

    def manifest(self):
        return dict(dataset=self.dataset, root=str(self.root), training=self.training,
                    skips=self.skips, voxel_bins=self.voxel_bins, samples=len(self),
                    legacy_event_time=self.legacy_event_time, event_coords=self.event_coords,
                    event_time_scale=self.event_time_scale, polarity=self.polarity,
                    scenes=[dict(name=s.path.name, root=str(s.path), frames=len(s.images),
                                 event_files=len(s.event_files), timestamp_source=s.timestamp_source,
                                 frame_time_first=float(s.frame_times[0]), frame_time_last=float(s.frame_times[-1]))
                            for s in self.scenes])

    def __getitem__(self, item):
        scene_index, start, skip = self.samples[item]
        scene = self.scenes[scene_index]
        if skip is None:
            candidates = [s for s in self.skips if start+s+1<len(scene.images)]
            skip = random.choice(candidates)
        end = start+skip+1
        target_indices = list(range(start+1,end))
        if self.training and self.num_targets is not None:
            target_indices = sorted(random.sample(target_indices, min(self.num_targets, len(target_indices))))
        elif self.gopro_odd_targets:
            # Matches upstream's zero-based odd intermediate index selection.
            target_indices = target_indices[1::2]
            if not target_indices:
                raise ValueError('GoPro odd-target protocol needs skip>=2')
        indices = [start]+target_indices+[end]
        images = [np.array(Image.open(scene.images[i]).convert('RGB'), dtype=np.float32)/255 for i in indices]
        height,width = images[0].shape[:2]
        if any(x.shape!=images[0].shape for x in images):
            raise ValueError(f'{scene.path}: image sizes change within interval')
        ch = cw = self.crop_size if self.training and self.crop_size else None
        if ch is not None:
            if ch>height or cw>width:
                raise ValueError('crop_size exceeds image size')
            y0,x0 = random.randint(0,height-ch),random.randint(0,width-cw)
        else:
            y0=x0=0; ch,cw=height,width
        images = [x[y0:y0+ch,x0:x0+cw] for x in images]
        tau0,tau1 = scene.frame_times[start],scene.frame_times[end]
        if scene.event_starts is None:
            files = scene.event_files[start:end]
        else:
            left = max(0,int(np.searchsorted(scene.event_starts,tau0,side='right'))-1)
            right = min(len(scene.event_files),int(np.searchsorted(scene.event_starts,tau1,side='left')))
            files = scene.event_files[left:right]
            if not files:
                raise ValueError('no event files overlap interval; check timestamp units and origin')
        parts=[]; raw_count=0; time_overlap=False
        for path in files:
            events = load_events(path,self.dataset,self.event_coords,self.polarity,self.event_time_scale)
            raw_count+=len(events)
            temporal=(events[:,2]>=tau0)&(events[:,2]<=tau1)
            time_overlap=time_overlap or bool(temporal.any())
            valid = ((events[:,0]>=x0)&(events[:,0]<x0+cw)&(events[:,1]>=y0)&(events[:,1]<y0+ch))
            if not self.legacy_event_time:
                valid &= temporal
            events=events[valid].copy()
            events[:,0]-=x0; events[:,1]-=y0
            parts.append(events)
        if not self.legacy_event_time and raw_count and not time_overlap:
            raise ValueError(f'{scene.path}: event files have no events within frame time bounds; check timestamp units and time origin')
        events=np.concatenate(parts) if parts else np.empty((0,4))
        voxel=events_to_voxel_grid(events,self.voxel_bins,ch,cw,tau0,tau1,self.legacy_event_time)
        packed=torch.from_numpy(np.stack(images)).permute(0,3,1,2).contiguous()
        if self.augmentation:
            for dim in (2,3):
                if random.random()<.5:
                    packed=packed.flip(dim);voxel=voxel.flip(dim-1)
        if self.legacy_event_time:
            times=torch.tensor([(i-start)/(end-start) for i in target_indices],dtype=torch.float32)
        else:
            times=torch.tensor([(scene.frame_times[i]-tau0)/(tau1-tau0) for i in target_indices],dtype=torch.float32)
        return dict(im0=packed[0],im1=packed[-1],targets=packed[1:-1],event=voxel,times=times,
                    scene=scene.path.name,start=start,skip=skip,
                    target_names=[scene.images[i].name for i in target_indices],
                    event_count=len(events))


def shared_time_collate(samples):
    """Reject incompatible batches instead of silently applying sample 0's times."""
    from torch.utils.data import default_collate
    if any(s['times'].shape!=samples[0]['times'].shape or not torch.allclose(s['times'],samples[0]['times'])
           for s in samples[1:]):
        raise ValueError('batch samples need identical target times; use batch_size=1 for mixed skips')
    return default_collate(samples)

