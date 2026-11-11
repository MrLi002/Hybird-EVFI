"""Interpolate a single keyframe pair from a full-interval voxel tensor."""
import argparse
from dataclasses import fields
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from models.config import ModelConfig
from util.options import add_model_arguments
from util.checkpoint import model_from_checkpoint
from util.runtime import resolve_device,pad_inputs
from util.event import load_events
from util.voxelization import events_to_voxel_grid


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_model_arguments(p,suppress_defaults=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--im0',required=True);p.add_argument('--im1',required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--voxel',help='.npy or .npz with voxel/event key, shape (128,H,W)')
    group.add_argument('--events',help='one raw x,y,t,p .npz containing all interval events')
    p.add_argument('--tau0',type=float);p.add_argument('--tau1',type=float)
    p.add_argument('--event_time_scale',type=float,default=1.)
    p.add_argument('--dataset',choices=['gopro','bsergb'],default='gopro')
    p.add_argument('--event_coords',choices=['pixels','ofaf'],default='pixels')
    p.add_argument('--polarity',choices=['auto','01','pm1'],default='auto')
    p.add_argument('--times',type=float,nargs='+')
    p.add_argument('--ratio',type=int,default=16)
    p.add_argument('--output',default='outputs/interpolation')
    p.add_argument('--device',default='auto');p.add_argument('--num_threads',type=int)
    args=p.parse_args(argv)
    if args.num_threads:torch.set_num_threads(args.num_threads)
    device=resolve_device(args.device)
    overrides={f.name:getattr(args,f.name) for f in fields(ModelConfig) if hasattr(args,f.name)}
    model,_=model_from_checkpoint(args.checkpoint,overrides,device)
    images=[torch.from_numpy(np.array(Image.open(path).convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None]
            for path in (args.im0,args.im1)]
    h,w=images[0].shape[-2:]
    if images[0].shape!=images[1].shape:raise ValueError('keyframe shapes differ')
    if args.events:
        if args.tau0 is None or args.tau1 is None:p.error('raw events require --tau0 and --tau1 in seconds')
        events=load_events(args.events,args.dataset,args.event_coords,args.polarity,args.event_time_scale)
        voxel=events_to_voxel_grid(events,model.voxel_bins,h,w,args.tau0,args.tau1)
    else:
        loaded=np.load(args.voxel,allow_pickle=False)
        if isinstance(loaded,np.lib.npyio.NpzFile):
            with loaded as data:
                key='voxel' if 'voxel' in data else 'event' if 'event' in data else data.files[0]
                array=np.array(data[key],copy=True)
        else:array=loaded
        voxel=torch.from_numpy(array).float()
    if voxel.shape!=(model.voxel_bins,h,w):raise ValueError('voxel must match configured bins and image size')
    times=args.times if args.times is not None else [j/args.ratio for j in range(1,args.ratio)]
    if not times or any(not 0<t<1 for t in times):raise ValueError('provide times in (0,1), or ratio>=2')
    (im0,im1,event),_=pad_inputs(images[0].to(device),images[1].to(device),voxel[None].to(device),model.size_multiple)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        cached=model.encode_pair(im0,im1,event)
        for index,t in enumerate(times):
            pred=model.render_time(cached,t)[0,:,:h,:w].clamp(0,1)
            array=(pred.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(array).save(output/f'{index:04d}_t{t:.8f}.png')
    (output/'times.json').write_text(json.dumps(times,indent=2))
    print(f'saved {len(times)} frames to {output}')


if __name__=='__main__':main()

