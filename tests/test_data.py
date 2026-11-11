"""Frame alignment, OF-AF sample layout and train/evaluation integration."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch

from datasets import GOPRO_Train, GOPRO_Test, BSERGB_Train, BSERGB_Test
from datasets.common import RawVFIData, shared_time_collate, DATA_PROTOCOL
from util.event import EventSequence
from util.voxelization import events_to_voxel_grid, to_voxel_grid


def make_scene(root, dataset='bsergb', frames=9, height=32, width=32, name='toy'):
    folder = root/name
    images, events = folder/'images', folder/'events'
    images.mkdir(parents=True); events.mkdir()
    for i in range(frames):
        # A frame index and spatial gradients detect RGB packing/cropping mistakes.
        yy, xx = np.mgrid[:height, :width]
        pixels = np.stack((np.full_like(xx, i), xx, yy), -1).astype(np.uint8)
        Image.fromarray(pixels).save(images/f'{i:05d}.png')
        if i < frames-1:
            x, y = np.array([5., 20.]), np.array([7., 21.])
            t = i*100+np.array([20., 70.])
            if dataset == 'bsergb':
                np.savez(events/f'{i:05d}.npz', x=(x+.25)*62000/1940,
                         y=(y+.25)*20160/630, timestamp=t, polarity=np.array([1, 0]))
            else:
                np.savez(events/f'{i:05d}.npz', x=x, y=y, t=t, p=np.array([1, -1]))
    return folder


def golden_voxel(first, last, bins, height=32, width=32):
    """Independent time-linear accumulation for known integer-coordinate fixtures."""
    result = np.zeros((bins, height, width), dtype=np.float32)
    start, end = first*100+20, (last-1)*100+70
    for index in range(first, last):
        for x, y, timestamp, sign in ((5, 7, index*100+20, 1), (20, 21, index*100+70, -1)):
            t = (timestamp-start)*(bins-1)/(end-start)
            left = int(np.floor(t))
            result[left, y, x] += sign*(1-(t-left))
            if left+1 < bins:
                result[left+1, y, x] += sign*(t-left)
    return torch.from_numpy(result)


@pytest.mark.parametrize('dataset,cls', [('bsergb', BSERGB_Train), ('gopro', GOPRO_Train)])
def test_ofaf_packed_tuple_and_exact_interval_without_timestamp_files(tmp_path, dataset, cls):
    folder = make_scene(tmp_path, dataset)
    # Even malformed timestamp text must be irrelevant to this loader.
    (folder/'timestamps.txt').write_text('not a timestamp')
    (folder/'images_timestamps.txt').write_text('also ignored')
    (folder/'events'/'notes.txt').write_text('not an event file')
    data = cls(128, tmp_path, 'toy', 3, crop_size=32, augmentation=False)
    imgs, voxel, times = data[1]
    assert len(data) == 5
    assert imgs.shape == (15, 32, 32)
    assert voxel.shape == (128, 32, 32)
    torch.testing.assert_close(times, torch.tensor([.25, .5, .75]))
    torch.testing.assert_close(imgs[:, 0, 0].reshape(5, 3)[:, 0], torch.arange(1, 6)/255)
    torch.testing.assert_close(voxel, golden_voxel(1, 5, 128), atol=1e-5, rtol=0)


def test_shared_crop_and_flip_keep_images_events_aligned(tmp_path, monkeypatch):
    make_scene(tmp_path, 'gopro', height=40, width=48)
    data = GOPRO_Train(8, tmp_path, 'toy', 1, crop_size=32)
    offsets = iter([4, 6])
    monkeypatch.setattr(random, 'randint', lambda *_: next(offsets))
    flips = iter([.1, .9])  # Vertical flip only.
    monkeypatch.setattr(random, 'uniform', lambda *_: next(flips))
    imgs, voxel, _ = data[0]
    assert imgs[1, 0, 0] == pytest.approx(6/255)
    assert imgs[2, 0, 0] == pytest.approx(35/255)
    # x=5 was cropped out; x=20,y=21 becomes x=14,y=17 and then y=14.
    support = voxel.abs().sum(0)
    assert support.count_nonzero() == 1 and support[14, 14] > 0
    # Remaining event times are 70 and 170: OF-AF normalizes after the crop.
    assert voxel[0, 14, 14] == -1 and voxel[-1, 14, 14] == -1


def test_skip_enumeration_and_dictionary_adapter(tmp_path):
    make_scene(tmp_path)
    data = RawVFIData(tmp_path, 'bsergb', skips=(1, 3), crop_size=32, augmentation=False)
    assert len(data) == (9-1-1)+(9-3-1)
    assert sum(data[i]['skip'] == 3 for i in range(len(data))) == 5
    sample = data[7]
    assert sample['targets'].shape == (3, 3, 32, 32)
    assert sample['target_names'] == ['00001.png', '00002.png', '00003.png']
    torch.testing.assert_close(sample['im0'][0], torch.zeros(32, 32))
    torch.testing.assert_close(sample['im1'][0], torch.full((32, 32), 4/255))
    assert shared_time_collate([sample])['event'].shape == (1, 128, 32, 32)
    assert data.manifest()['protocol'] == DATA_PROTOCOL
    capped = RawVFIData(tmp_path, 'bsergb', skips=(1,), num_targets=4, crop_size=32)
    assert capped[0]['targets'].shape[0] == 1


@pytest.mark.parametrize('dataset,cls', [('bsergb', BSERGB_Test), ('gopro', GOPRO_Test)])
def test_eval_stride_and_target_selection(tmp_path, dataset, cls):
    make_scene(tmp_path, dataset)
    direct = cls(8, tmp_path, 'toy', 3)
    assert direct.starts == [0, 3]
    assert direct[0][0].shape == (15, 32, 32)
    assert direct.get_batch(3)[0].shape == (1, 15, 32, 32)
    data = RawVFIData(tmp_path, dataset, skips=(3,), voxel_bins=8, training=False)
    sample = data[0]
    if dataset == 'gopro':
        assert sample['target_names'] == ['00002.png']
        torch.testing.assert_close(sample['times'], torch.tensor([.5]))
    else:
        assert sample['target_names'] == ['00001.png', '00002.png', '00003.png']
        torch.testing.assert_close(sample['times'], torch.tensor([.25, .5, .75]))
    assert data[1]['start'] == 3


def test_bsergb_bad_frames_follow_original_protocol(tmp_path):
    make_scene(tmp_path, frames=40, height=8, width=8, name='basket_09')
    data = BSERGB_Test(8, tmp_path, 'basket_09', 1)
    assert data.starts == [i for i in range(38) if i not in range(29, 35)]


@pytest.mark.parametrize('dataset,cls', [('bsergb', BSERGB_Train), ('gopro', GOPRO_Train)])
def test_empty_events_and_single_timestamp_are_finite(tmp_path, dataset, cls):
    folder = make_scene(tmp_path, dataset, frames=3)
    for path in (folder/'events').glob('*.npz'):
        with np.load(path) as source:
            values = {key: value[:0] for key, value in source.items()}
        np.savez(path, **values)
    data = cls(8, tmp_path, 'toy', 1, crop_size=32, augmentation=False)
    assert data[0][1].count_nonzero() == 0
    sequence = EventSequence(np.array([[2, 3, 100, 1.]]), 8, 8)
    voxel = to_voxel_grid(sequence, 8)
    assert torch.isfinite(voxel).all() and voxel[0, 3, 2] == 1


def test_high_rate_file_count_and_reset_timestamps_are_rejected(tmp_path):
    folder = make_scene(tmp_path, 'gopro', frames=3)
    np.savez(folder/'events'/'00002.npz', x=[1], y=[1], t=[200], p=[1])
    with pytest.raises(ValueError, match='frame-aligned'):
        GOPRO_Train(8, tmp_path, 'toy', 1, crop_size=32)
    (folder/'events'/'00002.npz').unlink()
    np.savez(folder/'events'/'00001.npz', x=[1, 2], y=[1, 2], t=[20, 70], p=[1, -1])
    data = GOPRO_Train(8, tmp_path, 'toy', 1, crop_size=32)
    with pytest.raises(ValueError, match='continuous time axis'):
        data[0]


def test_batch_rejects_inconsistent_times(tmp_path):
    make_scene(tmp_path)
    data = RawVFIData(tmp_path, 'bsergb', skips=(3,), crop_size=32)
    a = data[0]; b = dict(a); b['times'] = a['times']+.01
    with pytest.raises(ValueError, match='identical'):
        shared_time_collate([a, b])


def test_default_skips_and_removed_time_arguments():
    from train_model import parser
    args = parser().parse_args([])
    assert args.num_targets is None and args.skips is None
    for name in ('legacy_event_time', 'frame_timestamps', 'image_fps', 'time_offset',
                 'frame_time_scale', 'event_time_scale', 'event_coords', 'polarity',
                 'scene_file', 'eval_stride', 'gopro_odd_targets', 'include_bad_frames'):
        assert not hasattr(args, name)


def test_old_pipeline_resume_requires_weight_transfer(tmp_path):
    from train_model import main
    path = tmp_path/'old.pth'
    torch.save(dict(args={}, optimizer={}), path)
    with pytest.raises(ValueError, match='previous data pipeline'):
        main(['--resume', str(path)])


def test_low_level_explicit_time_voxelizer_remains_available():
    # Preserve the utility API used independently of dataset loading.
    events = np.array([[1., 2., .25, 1], [2., 1., .75, -1]])
    voxel = events_to_voxel_grid(events, 128, 4, 4, 0, 1)
    assert float(voxel[:64].sum()) == pytest.approx(1)
    assert float(voxel[64:].sum()) == pytest.approx(-1)
