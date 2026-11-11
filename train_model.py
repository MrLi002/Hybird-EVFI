"""Training entry point following the OF-AF directory and checkpoint layout."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import DATA_PROTOCOL, get_dataset, shared_time_collate
from models.model import HybridModel
from models.config import ModelConfig
from models.losses import HybridLoss
from util.options import add_model_arguments,add_data_arguments
from util.checkpoint import (load_checkpoint,load_matching_weights,state_dict_from_checkpoint,
    capture_rng,restore_rng,atomic_save)
from util.runtime import seed_everything,seed_worker,resolve_device,pad_inputs


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    add_model_arguments(p);add_data_arguments(p)
    p.add_argument('--epoch',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--eta_min',type=float,default=1e-7)
    p.add_argument('--old_lr_scale',type=float,default=.1)
    p.add_argument('--batch_size',type=int,default=1)
    p.add_argument('--num_worker',type=int,default=8)
    p.add_argument('--save_dir',default='train/hybrid')
    p.add_argument('--weights_path',default=None)
    p.add_argument('--only_weights',action='store_true',help='compatibility flag: weights_path always loads weights only')
    p.add_argument('--cbm_weights',default=None)
    p.add_argument('--strict_weights',action='store_true')
    p.add_argument('--resume',default=None)
    p.add_argument('--reset_scheduler',action='store_true',help='rebuild LR schedule when extending total epochs')
    p.add_argument('--w_lpips',type=float,default=.1)
    p.add_argument('--grad_clip',type=float,default=.1)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--device',default='auto')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--num_threads',type=int,default=None)
    p.add_argument('--no_augmentation',action='store_true')
    p.add_argument('--max_steps',type=int,default=None,help='debug cap per epoch; leave unset for training')
    p.add_argument('--log_every',type=int,default=20)
    p.add_argument('--val_root',default=None)
    p.add_argument('--val_every',type=int,default=1)
    return p


def explicit_destinations(p,argv):
    mapping={flag:action.dest for action in p._actions for flag in action.option_strings}
    return {mapping[token.split('=')[0]] for token in argv if token.split('=')[0] in mapping}


@torch.no_grad()
def validate(model,dataset,device):
    model.eval();scores=[]
    for sample in dataset:
        im0,im1,event=(sample[key][None].to(device) for key in ('im0','im1','event'))
        (im0,im1,event),(h,w)=pad_inputs(im0,im1,event,model.size_multiple)
        predictions=model(im0,im1,event,sample['times'][None].to(device))[0,:,:,:h,:w].clamp(0,1)
        targets=sample['targets'].to(device)
        mse=(predictions-targets).square().mean((1,2,3)).clamp_min(1e-12)
        scores.extend((-10*mse.log10()).cpu().tolist())
    model.train()
    return sum(scores)/len(scores)


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    p=parser();args=p.parse_args(argv)
    resume=load_checkpoint(args.resume) if args.resume else None
    if resume is not None:
        if 'args' not in resume or 'optimizer' not in resume:
            raise ValueError('--resume needs a training checkpoint; use --weights_path for author weights')
        if resume.get('data_protocol') != DATA_PROTOCOL:
            raise ValueError('checkpoint uses the previous data pipeline; use --weights_path '
                             'to start training with frame-aligned OF-AF data')
        explicit=explicit_destinations(p,argv)
        for key,value in resume['args'].items():
            if key not in explicit and hasattr(args,key):
                setattr(args,key,value)
        for key,value in resume['model_config'].items():
            if key in explicit and getattr(args,key)!=value and not (key=='basr_heads' and tuple(getattr(args,key))==tuple(value)):
                raise ValueError(f'cannot change model configuration {key} during resume')
            setattr(args,key,value)
        if args.epoch!=resume['args']['epoch'] and not args.reset_scheduler:
            raise ValueError('changing total epochs during resume requires --reset_scheduler')
    if not args.data_root:
        p.error('--data_root is required for new training')
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if args.epoch<1 or args.lr<=0 or args.batch_size<1 or args.val_every<1:
        raise ValueError('invalid training configuration')
    if args.log_every<1 or (args.max_steps is not None and args.max_steps<1):
        raise ValueError('log_every and debug max_steps must be positive')
    device=resolve_device(args.device)
    seed_everything(args.seed)
    cfg=ModelConfig.from_args(args)
    if args.crop_size% (32 if cfg.basr else 4):
        raise ValueError('crop_size must be divisible by 32 for BASR, otherwise by 4')
    dataset=get_dataset(args)
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_worker,
        pin_memory=device.type=='cuda',drop_last=True,worker_init_fn=seed_worker,
        generator=generator,collate_fn=shared_time_collate)
    if not len(loader):
        raise ValueError('dataset contains fewer samples than one training batch')
    val=None
    if args.val_root:
        val=get_dataset(args,training=False,root=args.val_root)
        training_paths={str(s.path.resolve()) for s in dataset.scenes}
        if training_paths & {str(s.path.resolve()) for s in val.scenes}:
            raise ValueError('training and validation scenes overlap')
    model=HybridModel(cfg).to(device)
    loaded_names=set();report={}
    if resume is not None:
        model.load_state_dict(state_dict_from_checkpoint(resume),strict=True)
    elif args.weights_path:
        checkpoint=load_checkpoint(args.weights_path)
        if args.strict_weights:
            model.load_state_dict(state_dict_from_checkpoint(checkpoint),strict=True)
            loaded_names=set(model.state_dict())
        else:
            loaded_names,report=load_matching_weights(model,checkpoint)
    if args.cbm_weights and resume is None:
        _,report['cbm']=load_matching_weights(model,load_checkpoint(args.cbm_weights),cbm=True)
    old,new=[],[]
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            (old if name in loaded_names else new).append(parameter)
    # Resume preserves saved group membership, including any original transfer-learning groups.
    if resume is not None:
        group_names=resume['optimizer_param_names']
        params=dict(model.named_parameters())
        groups=[dict(params=[params[name] for name in names],lr=args.lr) for names in group_names]
    else:
        groups=[]
        if old:groups.append(dict(params=old,lr=args.lr*args.old_lr_scale))
        if new:groups.append(dict(params=new,lr=args.lr))
    optimizer=torch.optim.Adam(groups,betas=(.9,.999))
    steps=min(len(loader),args.max_steps) if args.max_steps else len(loader)
    scheduler=CosineAnnealingLR(optimizer,T_max=args.epoch*steps,eta_min=args.eta_min)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda')
    start_epoch=0;global_step=0;best_psnr=float('-inf')
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        if not args.reset_scheduler:scheduler.load_state_dict(resume['scheduler'])
        if resume['scaler']:
            scaler.load_state_dict(resume['scaler'])
        start_epoch=resume['epoch']+1;global_step=resume['global_step'];best_psnr=resume.get('best_psnr',best_psnr)
        generator.set_state(resume['loader_rng'])
    criterion=HybridLoss(args.w_lpips,args.w_flow,args.w_ps,args.legacy_warp,args.legacy_lpips).to(device)
    if resume is not None:
        # LPIPS construction can consume RNG while creating its pretrained network.
        restore_rng(resume['rng'])
    directory=Path(args.save_dir);directory.mkdir(parents=True,exist_ok=True)
    (directory/'args.json').write_text(json.dumps(vars(args),ensure_ascii=False,indent=2))
    manifest=dataset.manifest()
    if val is not None:manifest['validation']=val.manifest()
    (directory/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if report:(directory/'weight_loading.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    names_by_id={id(parameter):name for name,parameter in model.named_parameters()}
    param_names=[[names_by_id[id(parameter)] for parameter in group['params']] for group in optimizer.param_groups]
    print(f'device={device}, samples={len(dataset)}, batches={len(loader)}, parameters={sum(p.numel() for p in model.parameters()):,}',flush=True)
    model.train()
    for epoch in range(start_epoch,args.epoch):
        sums={};begin=time.perf_counter()
        for index,batch in enumerate(loader):
            if index>=steps:break
            im0,im1,event,targets,times=(batch[key].to(device,non_blocking=True) for key in ('im0','im1','event','targets','times'))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
                outputs=model(im0,im1,event,times,return_aux=True)
                loss,parts=criterion(outputs,targets,im0,im1)
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at epoch {epoch}, batch {index}')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old_scale:scheduler.step()
            global_step+=1
            for key,value in parts.items():sums[key]=sums.get(key,0.)+float(value)
            if index%args.log_every==0:
                print(f'epoch={epoch} batch={index}/{steps} loss={float(loss):.6f} lr={optimizer.param_groups[-1]["lr"]:.3e}',flush=True)
        record=dict(epoch=epoch,global_step=global_step,seconds=time.perf_counter()-begin,
                    **{key:value/steps for key,value in sums.items()})
        improved=False
        if val is not None and (epoch+1)%args.val_every==0:
            record['val_psnr_float']=validate(model,val,device)
            improved=record['val_psnr_float']>best_psnr
            best_psnr=max(best_psnr,record['val_psnr_float'])
        checkpoint=dict(model=model.state_dict(),model_config=asdict(cfg),args=vars(args),data_protocol=DATA_PROTOCOL,
            optimizer=optimizer.state_dict(),optimizer_param_names=param_names,scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),epoch=epoch,global_step=global_step,best_psnr=best_psnr,
            rng=capture_rng(),loader_rng=generator.get_state())
        atomic_save(checkpoint,directory/'last.pth');atomic_save(checkpoint,directory/f'epoch{epoch}.pth')
        if improved:atomic_save(checkpoint,directory/'best.pth')
        with (directory/'log.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
