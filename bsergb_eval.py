"""BS-ERGB/GoPro evaluation, with explicit metric and target-selection protocols."""
import argparse
from dataclasses import fields
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
import torch
from datasets.common import RawVFIData
from models.config import ModelConfig
from util.options import add_model_arguments,add_data_arguments
from util.checkpoint import model_from_checkpoint
from util.runtime import resolve_device,pad_inputs
from util.metrics import ImageMetrics


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_model_arguments(p,suppress_defaults=True);add_data_arguments(p,evaluation=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--save_root',default='eval/hybrid')
    p.add_argument('--save_pred',action='store_true')
    p.add_argument('--skip_lpips',action='store_true')
    p.add_argument('--device',default='auto')
    p.add_argument('--num_threads',type=int,default=None)
    p.add_argument('--max_intervals',type=int,default=None,help='debug subset; leave unset for reported evaluation')
    args=p.parse_args(argv)
    if not args.data_root:p.error('--data_root/--test_root is required')
    if args.num_threads:torch.set_num_threads(args.num_threads)
    device=resolve_device(args.device)
    overrides={f.name:getattr(args,f.name) for f in fields(ModelConfig) if hasattr(args,f.name)}
    model,checkpoint=model_from_checkpoint(args.checkpoint,overrides,device)
    # Frame-aligned files and event-bound normalization match OF-AF BS-ERGB.
    skips=args.skips or ([1,3] if args.dataset=='bsergb' else [7,15])
    dataset=RawVFIData(args.data_root,dataset=args.dataset,skips=skips,voxel_bins=model.voxel_bins,
        num_targets=None,training=False,crop_size=None)
    metrics=ImageMetrics(device,not args.skip_lpips,args.legacy_lpips)
    directory=Path(args.save_root);directory.mkdir(parents=True,exist_ok=True)
    (directory/'data_manifest.json').write_text(json.dumps(dataset.manifest(),ensure_ascii=False,indent=2))
    (directory/'eval_args.json').write_text(json.dumps(vars(args),ensure_ascii=False,indent=2))
    rows=[];elapsed=0.;frames=0
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(),(directory/'per_frame.jsonl').open('w') as log:
        for interval,sample in enumerate(dataset):
            if args.max_intervals is not None and interval>=args.max_intervals:break
            im0,im1,event=(sample[key][None].to(device) for key in ('im0','im1','event'))
            (im0,im1,event),(height,width)=pad_inputs(im0,im1,event,model.size_multiple)
            if device.type=='cuda':torch.cuda.synchronize(device)
            begin=time.perf_counter();cached=model.encode_pair(im0,im1,event)
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed+=time.perf_counter()-begin
            for j,target_time in enumerate(sample['times']):
                if device.type=='cuda':torch.cuda.synchronize(device)
                begin=time.perf_counter()
                prediction=model.render_time(cached,float(target_time))[:,:,:height,:width].clamp(0,1)
                if device.type=='cuda':torch.cuda.synchronize(device)
                elapsed+=time.perf_counter()-begin;frames+=1
                target=sample['targets'][j:j+1].to(device)
                row=dict(scene=sample['scene'],skip=sample['skip'],start=sample['start'],
                         target=sample['target_names'][j],t=float(target_time),**metrics(prediction,target))
                rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                if args.save_pred:
                    dest=directory/str(sample['skip'])/sample['scene']/f"{sample['start']:06d}"
                    dest.mkdir(parents=True,exist_ok=True)
                    img=(prediction[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                    Image.fromarray(img).save(dest/sample['target_names'][j])
            print(f'{interval+1}/{len(dataset)} {sample["scene"]} skip={sample["skip"]}',flush=True)
            del cached
    if not rows:raise ValueError('no frames evaluated')
    metric_names=['psnr','ssim']+(['lpips'] if not args.skip_lpips else [])
    groups={}
    for skip in skips:
        selected=[row for row in rows if row['skip']==skip]
        if selected:groups[str(skip)]={name:float(np.mean([row[name] for row in selected])) for name in metric_names}
    summary=dict(frames=frames,by_skip=groups,
        overall={name:float(np.mean([row[name] for row in rows])) for name in metric_names},
        model_seconds=elapsed,seconds_per_frame=elapsed/frames,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()),
        timings_include='pair encoding plus per-target rendering; excludes IO, saving and image metrics; no warmup excluded',
        subset=args.max_intervals is not None,legacy_lpips=args.legacy_lpips,
        note='FloLPIPS is not implemented in this package; PSNR/SSIM/LPIPS are reported explicitly')
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
