"""Generate tiny deterministic raw-event scenes for interface checks, not research results."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def create(root,dataset='bsergb',frames=7,size=32):
    folder=Path(root)/'toy'
    images=folder/'images'
    events=folder/'events'
    images.mkdir(parents=True,exist_ok=True);events.mkdir(exist_ok=True)
    rng=np.random.default_rng(1234)
    timestamps=np.arange(frames,dtype=np.float64)/240
    for i in range(frames):
        image=rng.integers(0,256,(size,size,3),dtype=np.uint8)
        Image.fromarray(image).save(images/f'{i:05d}.png')
        if i<frames-1:
            count=128
            x=rng.integers(0,size,count).astype(np.float64)
            y=rng.integers(0,size,count).astype(np.float64)
            if dataset=='bsergb':
                x=(x+.25)*62000/1940
                y=(y+.25)*20160/630
            values=dict(x=x,y=y)
            t=timestamps[i]+rng.uniform(.01,.99,count)/240
            p=rng.integers(0,2,count,dtype=np.int8)
            if dataset=='bsergb':values.update(timestamp=t,polarity=p)
            else:values.update(t=t,p=2*p-1)
            np.savez(events/f'{i:05d}.npz',**values)
    return folder


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--dataset',choices=['bsergb','gopro'],default='bsergb')
    p.add_argument('--frames',type=int,default=7);p.add_argument('--size',type=int,default=32)
    args=p.parse_args();print(create(args.output,args.dataset,args.frames,args.size))
