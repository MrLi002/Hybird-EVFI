import random
import numpy as np
from PIL import Image
import pytest
import torch
from datasets.common import RawVFIData,shared_time_collate
from util.voxelization import events_to_voxel_grid


def make_scene(root,dataset='bsergb',frames=7,height=32,width=32,scaled=False):
    folder=root/'toy'
    images=folder/('images' if dataset=='bsergb' else 'imgs')
    events=folder/('events' if dataset=='bsergb' else 'events_clean_8x_v2')
    images.mkdir(parents=True);events.mkdir()
    rng=np.random.default_rng(123)
    times=np.arange(frames,dtype=np.float64)/240
    np.savetxt(folder/'images_timestamps.txt',times*(1000 if scaled else 1),fmt='%.12f')
    for i in range(frames):
        Image.fromarray(rng.integers(0,256,(height,width,3),dtype=np.uint8)).save(images/f'{i:05d}.png')
        if i<frames-1:
            t=(times[i]+np.array([.2,.7])/240)*(1e6 if scaled else 1)
            values=dict(x=np.array([5.25,20.5]),y=np.array([7.5,21.25]))
            if dataset=='bsergb':values.update(timestamp=t,polarity=np.array([1,0]))
            else:values.update(t=t,p=np.array([1,-1]))
            np.savez(events/f'{i:05d}.npz',**values)
    return folder


def test_voxel_mass_time_bounds_empty_single_and_crop_time_invariance():
    events=np.array([[1.5,2.25,.25,1],[2.,1.,.75,-1]],dtype=np.float64)
    voxel=events_to_voxel_grid(events,128,4,4,0,1)
    assert voxel.shape==(128,4,4)
    assert float(voxel.sum())==pytest.approx(0,abs=1e-6)
    assert float(voxel[:64].sum())==pytest.approx(1)
    assert float(voxel[64:].sum())==pytest.approx(-1)
    assert events_to_voxel_grid(np.empty((0,4)),128,4,4,0,1).count_nonzero()==0
    single=events_to_voxel_grid(events[:1],128,4,4,0,1)
    torch.testing.assert_close(single[:,1:4,:],voxel[:,1:4,:]-events_to_voxel_grid(events[1:],128,4,4,0,1)[:,1:4,:])
    assert torch.isfinite(single).all()
    # Cropping changes x/y but must not renormalize the surviving event time.
    shifted=events[:1].copy();shifted[:,0]-=1
    cropped=events_to_voxel_grid(shifted,128,4,3,0,1)
    torch.testing.assert_close(single.sum((1,2)),cropped.sum((1,2)))


@pytest.mark.parametrize('dataset',['bsergb','gopro'])
def test_raw_dataset_target_sampling_and_physical_units(tmp_path,dataset):
    make_scene(tmp_path,dataset,scaled=True)
    data=RawVFIData(tmp_path,dataset,skips=(1,3),num_targets=4,crop_size=32,
                    event_coords='pixels',frame_time_scale=1e-3,event_time_scale=1e-6)
    random.seed(7)
    sample=data[0]
    assert sample['event'].shape==(128,32,32)
    assert sample['targets'].shape[0] in (1,3)
    assert torch.all((sample['times']>0)&(sample['times']<1))
    assert sample['times'].tolist()==sorted(sample['times'].tolist())
    batch=shared_time_collate([sample]);assert batch['event'].shape==(1,128,32,32)
    assert data.manifest()['scenes'][0]['frames']==7


def test_missing_frame_times_and_legacy_opt_in(tmp_path):
    folder=make_scene(tmp_path)
    (folder/'images_timestamps.txt').unlink()
    with pytest.raises(ValueError,match='timestamps missing'):
        RawVFIData(tmp_path,'bsergb',skips=(1,),event_coords='pixels')
    data=RawVFIData(tmp_path,'bsergb',skips=(1,),crop_size=32,event_coords='pixels',legacy_event_time=True)
    assert data[0]['targets'].shape==(1,3,32,32)


def test_timestamp_mismatch_does_not_silently_zero_events(tmp_path):
    make_scene(tmp_path,scaled=True)
    data=RawVFIData(tmp_path,'bsergb',skips=(1,),crop_size=32,event_coords='pixels')
    with pytest.raises(ValueError,match='time bounds'):data[0]


def test_batch_rejects_inconsistent_times(tmp_path):
    make_scene(tmp_path)
    data=RawVFIData(tmp_path,'bsergb',skips=(3,),crop_size=32,event_coords='pixels')
    a=data[0];b=dict(a);b['times']=a['times']+.01
    with pytest.raises(ValueError,match='identical'):shared_time_collate([a,b])

