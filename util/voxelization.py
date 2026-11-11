"""Trilinear polarity accumulation using OF-AF's x,y,t,p convention."""
import numpy as np
import torch


def events_to_voxel_grid(events, bins, height, width, start_time=None, end_time=None, legacy=False):
    if bins < 2:
        raise ValueError('at least two time bins are required')
    events = np.asarray(events, dtype=np.float64).reshape(-1, 4)
    voxel = torch.zeros(bins*height*width, dtype=torch.float32)
    if len(events) == 0:
        return voxel.reshape(bins, height, width)
    if not np.isfinite(events).all():
        raise ValueError('event coordinates, times and polarities must be finite')
    if legacy:
        start_time, end_time = events[:, 2].min(), events[:, 2].max()
        if end_time == start_time:
            # Upstream divides by zero for this case. An empty normalized duration carries no temporal ordering.
            end_time = start_time+1.
    if start_time is None or end_time is None or end_time <= start_time:
        raise ValueError('explicit increasing keyframe time bounds are required')
    selected = (events[:, 2] >= start_time) & (events[:, 2] <= end_time)
    events = events[selected]
    if not len(events):
        return voxel.reshape(bins, height, width)
    x = torch.from_numpy(events[:, 0]).float()
    y = torch.from_numpy(events[:, 1]).float()
    # Subtract time origin in float64 before converting to float32 (microsecond timestamps).
    t = torch.from_numpy((events[:, 2]-start_time)*(bins-1)/(end_time-start_time)).float()
    polarity = torch.from_numpy(events[:, 3]).float()
    lx, ly, lt = x.floor(), y.floor(), t.floor()
    for xi in (lx, lx+1):
        for yi in (ly, ly+1):
            for ti in (lt, lt+1):
                valid = (xi>=0)&(xi<width)&(yi>=0)&(yi<height)&(ti>=0)&(ti<bins)
                index = (xi[valid]+yi[valid]*width+ti[valid]*width*height).long()
                weight = polarity*(1-(xi-x).abs())*(1-(yi-y).abs())*(1-(ti-t).abs())
                voxel.index_add_(0, index, weight[valid])
    return voxel.reshape(bins, height, width)


def to_voxel_grid(event_sequence, nb_of_time_bins=128, remapping_maps=None):
    features = np.array(event_sequence._features, copy=True)
    if remapping_maps is not None:
        maps = np.asarray(remapping_maps)
        features[:, :2] = maps[:, features[:, 1].astype(int), features[:, 0].astype(int)].T
    return events_to_voxel_grid(features, nb_of_time_bins, event_sequence._image_height,
                                event_sequence._image_width, event_sequence.start_time(), event_sequence.end_time())

