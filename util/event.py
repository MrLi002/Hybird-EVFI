"""Small OF-AF-compatible event sequence; raw events are always x,y,t,p."""
from pathlib import Path
import numpy as np


def load_events(path, dataset='gopro', coords='ofaf', polarity='auto', time_scale=1.):
    with np.load(path, allow_pickle=False) as data:
        def get(*keys):
            for key in keys:
                if key in data:
                    return np.asarray(data[key]).reshape(-1).astype(np.float64)
            raise KeyError(f'{path}: expected one of {keys}')
        x, y = get('x'), get('y')
        t, p = get('timestamp', 't', 'timestamps'), get('polarity', 'p')
    if not (len(x)==len(y)==len(t)==len(p)):
        raise ValueError(f'{path}: event columns have different lengths')
    if dataset == 'bsergb' and coords == 'ofaf':
        x = np.floor(2*970*x/62000)
        y = np.floor(630*y/20160)
    elif coords not in ('ofaf', 'pixels'):
        raise ValueError('event coordinate mode must be ofaf or pixels')
    if polarity == '01' or (polarity == 'auto' and len(p) and np.all((p==0)|(p==1))):
        p = 2*p-1
    if not np.all((p==-1)|(p==1)):
        raise ValueError(f'{path}: polarities must encode -1/+1 or 0/1')
    return np.column_stack((x, y, t*time_scale, p))


class EventSequence:
    def __init__(self, features, image_height, image_width, start_time=None, end_time=None):
        self._features = np.asarray(features, dtype=np.float64).reshape(-1, 4)
        self._image_height, self._image_width = image_height, image_width
        self._start_time = float(start_time) if start_time is not None else (self._features[:, 2].min() if len(self._features) else 0.)
        self._end_time = float(end_time) if end_time is not None else (self._features[:, 2].max() if len(self._features) else 0.)

    def start_time(self): return self._start_time
    def end_time(self): return self._end_time
    def duration(self): return self._end_time-self._start_time
    def __len__(self): return len(self._features)

    @classmethod
    def from_npz_files(cls, paths, image_height, image_width, start_time=None, end_time=None,
                       bsergb=False, size=None, **kwargs):
        options = dict(coords='ofaf', polarity='01' if bsergb else 'pm1')
        options.update(kwargs)
        parts = [load_events(path, dataset='bsergb' if bsergb else 'gopro', **options) for path in paths]
        previous_end = None
        for path, part in zip(paths, parts):
            if len(part):
                begin, end = part[:, 2].min(), part[:, 2].max()
                if previous_end is not None and begin < previous_end:
                    raise ValueError(f'{path}: event files must share a continuous time axis; '
                                     'per-file timestamp resets must be offset before loading')
                previous_end = end
        events = np.concatenate(parts) if parts else np.empty((0, 4))
        if size is not None:
            y0, x0, y1, x1 = size
            events = events[(events[:,0]>=x0)&(events[:,0]<x1)&(events[:,1]>=y0)&(events[:,1]<y1)].copy()
            events[:, 0] -= x0; events[:, 1] -= y0
        if not len(events):
            return None
        return cls(events, image_height, image_width, start_time, end_time)
