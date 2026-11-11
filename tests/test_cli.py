"""Small CPU CLI integration checks; no pretrained weights or real datasets needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from scripts.create_toy_dataset import create
from util.checkpoint import load_checkpoint

ROOT=Path(__file__).resolve().parents[1]


def run(*args):
    result=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=os.environ.copy(),timeout=180)
    assert result.returncode==0,result.stdout
    return result.stdout


def test_train_resume_reproduces_next_epoch_and_eval_inference(tmp_path):
    data=tmp_path/'data';create(data)
    first=tmp_path/'first';resumed=tmp_path/'resumed'
    run('train_model.py','--dataset','bsergb','--data_root',data,'--event_coords','pixels',
        '--skips','1','--refine','resblock','--voxel_bins','8','--nb_of_flow','2',
        '--crop_size','32','--num_targets','1','--num_worker','0','--epoch','2',
        '--max_steps','1','--w_lpips','0','--device','cpu','--num_threads','2','--save_dir',first)
    run('train_model.py','--resume',first/'epoch0.pth','--save_dir',resumed)
    a,b=load_checkpoint(first/'epoch1.pth'),load_checkpoint(resumed/'epoch1.pth')
    assert a['global_step']==b['global_step']==2
    assert a['scheduler']==b['scheduler']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    evaluation=tmp_path/'eval'
    run('bsergb_eval.py','--checkpoint',resumed/'last.pth','--test_root',data,
        '--event_coords','pixels','--skips','1','--skip_lpips','--max_intervals','1',
        '--device','cpu','--num_threads','2','--save_root',evaluation,'--save_pred')
    summary=json.loads((evaluation/'summary.json').read_text())
    assert summary['frames']==1 and summary['subset'] and '1' in summary['by_skip']
    assert list(evaluation.rglob('*.png'))
    voxel=tmp_path/'voxel.npy';np.save(voxel,np.zeros((8,32,32),np.float32))
    output=tmp_path/'interpolation'
    run('interpolate.py','--checkpoint',resumed/'last.pth','--voxel',voxel,
        '--im0',data/'toy/images/00000.png','--im1',data/'toy/images/00002.png',
        '--times','.3','.5','--device','cpu','--num_threads','2','--output',output)
    assert len(list(output.glob('*.png')))==2


def test_large_all_modules_cli_backward(tmp_path):
    data=tmp_path/'data';create(data)
    output=tmp_path/'all_modules'
    run('train_model.py','--dataset','bsergb','--data_root',data,'--event_coords','pixels',
        '--skips','1','--refine','transformer','--unit_dim','44','--voxel_bins','8','--nb_of_flow','2',
        '--basr','--mask_in_refine','--flow_corr','--splat_metric','--w_flow','.1','--w_ps','.1',
        '--checkpoint_bifeb','--checkpoint_refine','--crop_size','32','--num_targets','1',
        '--num_worker','0','--epoch','1','--max_steps','1','--w_lpips','0',
        '--device','cpu','--num_threads','2','--save_dir',output)
    checkpoint=load_checkpoint(output/'last.pth')
    assert checkpoint['model_config']['unit_dim']==44 and checkpoint['model_config']['basr']
    assert checkpoint['model']['biw.fuse_warp.metric_head.weight'].abs().sum()>0
    assert checkpoint['model']['flow_corr.decoder.4.weight'].abs().sum()>0

