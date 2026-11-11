from dataclasses import asdict
from pathlib import Path
import os
import random
import numpy as np
import torch
from models.config import ModelConfig
from models.model import HybridModel


def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location, weights_only=False)


def state_dict_from_checkpoint(checkpoint):
    state = checkpoint
    for key in ('model','state_dict','net','params'):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError('checkpoint does not contain a model state dictionary')
    return {key.removeprefix('module.'):value for key,value in state.items()}


def load_matching_weights(model, checkpoint, cbm=False):
    source = state_dict_from_checkpoint(checkpoint)
    target = model.state_dict()
    matched, skipped = {}, []
    for key,value in source.items():
        mapped = key
        if cbm:
            if key.startswith('transformer.'):
                mapped = 'refine_head.'+key
                if any(name in key for name in ('last_conv0','last_conv1','last_conv2',
                        'enc_conv0','build_kv0_warp','build_kv1_warp','build_kv2_warp')):
                    skipped.append(key); continue
            elif key.startswith('encoder_f.'):
                mapped = 'refine_head.'+key
            elif key.startswith('encoder_e.'):
                mapped = 'event_adapter.enc.'+key[len('encoder_e.'):]
            else:
                skipped.append(key); continue
        if mapped in target and hasattr(value,'shape') and value.shape==target[mapped].shape:
            matched[mapped] = value
        else:
            skipped.append(key)
    if not matched:
        raise ValueError('no parameter names/shapes match; check model width and checkpoint type')
    incompatible = model.load_state_dict(matched, strict=False)
    return set(matched), dict(loaded=len(matched), missing=incompatible.missing_keys, skipped=skipped)


def capture_rng():
    result=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        result['cuda']=torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(checkpoint, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(checkpoint,temporary)
    os.replace(temporary,path)


def model_from_checkpoint(path, overrides=None, device='cpu'):
    checkpoint=load_checkpoint(path)
    config=dict(checkpoint.get('model_config',{})) if isinstance(checkpoint,dict) else {}
    config.update(overrides or {})
    model=HybridModel(ModelConfig.from_args(config))
    model.load_state_dict(state_dict_from_checkpoint(checkpoint),strict=True)
    return model.to(device).eval(),checkpoint

