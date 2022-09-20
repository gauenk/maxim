import math
import torch as th
import torch.nn as nn
import os
import copy
ccopy = copy.copy
from einops import repeat
from pathlib import Path
import collections
from collections import OrderedDict

import io
import numpy as np
import tensorflow as tf

from .model_keys import translate_attn_mode,expand_attn_mode
from .qkv_convert import qkv_convert_state,block_name2num

def get_params(ckpt_path):
    """Get params checkpoint."""

    with tf.io.gfile.GFile(ckpt_path, 'rb') as f:
        data = f.read()

    values = np.load(io.BytesIO(data))
    params = recover_tree(*zip(*values.items()))
    params = params['opt']['target']

    return params

def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.
  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.
  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    mpath = "model_epoch_{}_{}.pth".format(epoch,session)
    model_out_path = os.path.join(model_dir,mpath)
    th.save(state, model_out_path)


def load_checkpoint_qkv(model, weights,in_attn_modes, out_attn_modes,
                        prefix="module.",reset=False):
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = qkv_convert_state(
        state_dict,in_attn_modes,out_attn_modes,
        prefix=prefix,reset=reset)
    model.load_state_dict(new_state_dict)

def load_checkpoint_module(model, weights):
    checkpoint = th.load(weights)
    try:
        # model.load_state_dict(checkpoint["state_dict"])
        raise ValueError("")
    except Exception as e:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_multigpu(model, weights):
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = th.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = th.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'Uformer':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    elif arch == 'Uformer16':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff')
    elif arch == 'Uformer32':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff')
    elif arch == 'Uformer_CatCross':
        model_restoration = Uformer_CatCross(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=8,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    elif arch == 'Uformer_Cross':
        model_restoration = Uformer_Cross(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    else:
        raise Exception("Arch error!")

    return model_restoration

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]

def temporal_chop(x,tsize,fwd_fxn,flows=None):
    nframes = x.shape[0]
    nslice = (nframes-1)//tsize+1
    x_agg = []
    for ti in range(nslice):
        ts = ti*tsize
        te = min((ti+1)*tsize,nframes)
        tslice = slice(ts,te)
        if flows:
            x_t = fwd_fxn(x[tslice],flows)
        else:
            x_t = fwd_fxn(x[tslice])
        x_agg.append(x_t)
    x_agg = th.cat(x_agg)
    return x_agg


def expand2square(timg,factor=16.0):
    t, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = th.zeros(t,3,X,X).type_as(timg) # 3, h,w
    mask = th.zeros(t,1,X,X).type_as(timg)

    print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Loading Checkpoint by Filename of Substr
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def load_checkpoint(model,use_train,substr="",croot="output/checkpoints/"):
    # -- do we load --
    load = use_train == "true" or use_train is True

    # -- load to model --
    if load:
        mpath = load_recent(croot,substr)
        print("Loading Model Checkpoint: ",mpath)
        state = th.load(mpath)['state_dict']
        remove_lightning_load_state(state)
        model.load_state_dict(state)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Loading the Most Recent File
#
#     think:
#     "path/to/dir/","this-is-my-uuid"
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def load_recent(root,substr):
    root = Path(root)
    if not root.exists():
        raise ValueError(f"Load directory [{root}] does not exist.")
    files = []
    for fn in root.iterdir():
        fn = str(fn)
        if substr == "" or substr in fn:
            files.append(fn)
    files = sorted(files,key=os.path.getmtime)
    files = [f for f in reversed(files)]

    # -- error --
    if len(files) == 0:
        raise ValueError(f"Unable to file any files with substr [{substr}]")

    return str(files[0])

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Modifying Layers In-Place after Loading
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def reset_product_attn_mods(model):
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            submod = model
            submods = name.split(".")
            for submod_i in submods:
                submod = getattr(submod,submod_i)
            submod.data = th.randn_like(submod.data).clamp(-1,1)/100.

def filter_rel_pos(model,in_attn_mode):
    attn_modes = expand_attn_mode(in_attn_mode)
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            bname = name.split(".")[0]
            bnum = block_name2num(bname)
            attn_mode_b = attn_modes[bnum]
            if attn_mode_b == "product_dnls":
                submod = model
                submods = name.split(".")
                for submod_i in submods[:-1]:
                    submod = getattr(submod,submod_i)
                setattr(submod,submods[-1],None)

def get_product_attn_params(model):
    params = []
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            # delattr(model,name)
            param.requires_grad_(False)
            continue
        params.append(param)
    return params

def apply_freeze(model,freeze):
    if freeze is False: return
    unset_names = []
    for name,param in model.named_parameters():
        # print(name)
        bname = name.split(".")[0]
        bnum = block_name2num(bname)
        if bnum == -1: unset_names.append(name)
        freeze_b = freeze[bnum]
        if freeze_b is True:
            param.requires_grad_(False)
    # print(unset_names)

# -- embed dims --
def expand_embed_dims(attn_modes,embed_dim_w,embed_dim_pd):
    exp_embed_dims = []
    for attn_mode in attn_modes:
        # print("attn_mode: ",attn_mode)
        if "window" in attn_mode:
            exp_embed_dims.append(embed_dim_w)
        elif "product" in attn_mode:
            exp_embed_dims.append(embed_dim_pd)
        else:
            raise ValueError(f"Uknown attn_mode [{attn_mode}]")
    assert len(exp_embed_dims) == 5
    return exp_embed_dims
