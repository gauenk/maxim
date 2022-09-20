"""

Test maxim outout for denoising an SIDD image

"""


# -- misc --
import sys,tqdm,pytest,math,random
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- vision --
import jax
import flax
import tensorflow as tf
import jax.numpy as jnp

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import dnls # supporting
from torchvision.transforms.functional import center_crop

# -- package imports [to test] --
import maxim
from maxim.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats
from maxim.utils.misc import rslice_pair

# -- check if reordered --
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_denose_rgb/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"sigma":[50.],"ref_version":["ref","original"]}
    test_lists = {"sigma":[50.],"ref_version":["ref"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def weird_unlist(preds):
    if isinstance(preds, list):
        preds = preds[-1]
        if isinstance(preds, list):
            preds = preds[-1]
    return preds

def test_fwd(sigma):
    # -- params --
    device = "cuda:0"
    # vid_set = "sidd_rgb"
    # vid_name = "00"
    # dset = "val"
    vid_set = "set8"
    vid_name = "motorbike"
    verbose = False
    isize = "128_128"
    dset = "te"
    flow = False
    noise_version = "blur"

    # -- timer --
    timer = maxim.utils.timer.ExpTimer()

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = vid_name
    cfg.isize = isize
    cfg.sigma = sigma
    attn_mode = "product_dnls"

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name == g]
    index = indices[0]

    # -- unpack --
    sample = data[dset][index]
    region = sample['region']
    region[0],region[1] = 0,2
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = rslice_pair(noisy,clean,region)
    noisy,clean = noisy.cpu().numpy(),clean.cpu().numpy()
    noisy = rearrange(noisy,'t c h w -> t h w c')
    clean = rearrange(clean,'t c h w -> t h w c')
    vid_frames = sample['fnums']
    noisy /= 255.
    # print("noisy.shape: ",noisy.shape)

    # -- flows --
    t,c,h,w = noisy.shape
    flows = edict()
    flows.fflow = np.zeros((t,h,w,2))
    flows.bflow = np.zeros((t,h,w,2))

    # -- original exec --
    model_gt,params_gt = maxim.load_model(sigma,attn_mode="original")
    deno_gt = model_gt.apply({'params': flax.core.freeze(params_gt)}, noisy)
    deno_gt = weird_unlist(deno_gt)

    # -- refactored exec --
    model_te,params_te = maxim.load_model(sigma,attn_mode="product")
    deno_te = model_te.apply({'params': flax.core.freeze(params_te)}, noisy)
    deno_te = weird_unlist(deno_te)

    # -- viz --
    print(timer)
    if verbose:
        print(deno_gt[0,0,:3,:3])
        print(deno_te[0,0,:3,:3])

    # -- viz --
    # diff_s = np.abs(deno_gt - deno_te)# / (deno_gt.abs()+1e-5)
    # print(diff_s.max())
    # diff_s /= diff_s.max()
    # print("diff_s.shape: ",diff_s.shape)
    # dnls.testing.data.save_burst(diff_s[:3],SAVE_DIR,"diff")
    # dnls.testing.data.save_burst(deno_gt[:3],SAVE_DIR,"deno_gt")
    # dnls.testing.data.save_burst(deno_te[:3],SAVE_DIR,"deno_te")

    # -- test --
    error = np.abs(deno_gt - deno_te).mean().item()
    print(error)
    if verbose: print("error: ",error)
    assert error < 1e-5

def test_bwd(sigma):

    # -- params --
    device = "cuda:0"
    # vid_set = "sidd_rgb"
    # vid_name = "00"
    # dset = "val"
    vid_set = "set8"
    vid_name = "motorbike"
    verbose = False
    isize = "128_128"
    dset = "te"
    flow = False
    noise_version = "blur"

    # -- timer --
    timer = maxim.utils.timer.ExpTimer()

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = vid_name
    cfg.isize = isize
    cfg.sigma = sigma
    attn_mode = "product_dnls"

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name == g]
    index = indices[0]

    # -- unpack --
    sample = data[dset][index]
    region = sample['region']
    region[0],region[1] = 0,1
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = rslice_pair(noisy,clean,region)
    noisy,clean = noisy.cpu().numpy(),clean.cpu().numpy()
    noisy = rearrange(noisy,'t c h w -> t h w c')
    clean = rearrange(clean,'t c h w -> t h w c')
    vid_frames = sample['fnums']
    noisy /= 255.
    # print("noisy.shape: ",noisy.shape)

    # -- flows --
    t,c,h,w = noisy.shape
    flows = edict()
    flows.fflow = np.zeros((t,h,w,2))
    flows.bflow = np.zeros((t,h,w,2))

    # -- original exec --
    model_gt,params_gt = maxim.load_model(sigma,attn_mode="original")
    # @jax.jit
    def _loss_gt(_params, x, y):
        pred = model_gt.apply({'params': _params},x)
        pred = weird_unlist(pred)
        return jnp.mean((pred-y)**2)
    loss_gt = jax.value_and_grad(_loss_gt)
    _,grad_gt = loss_gt(params_gt,noisy,clean)
    grad_gt = maxim.utils.flatten_util.ravel_pytree(grad_gt)[0]

    # -- refactored exec --
    model_te,params_te = maxim.load_model(sigma,attn_mode="product")
    # @jax.jit
    def _loss_te(_params, x, y):
        pred = model_te.apply({'params': _params},x)
        pred = weird_unlist(pred)
        return jnp.mean((pred-y)**2)
    loss_te = jax.value_and_grad(_loss_te)
    _,grad_te = loss_te(params_te,noisy,clean)
    grad_te = maxim.utils.flatten_util.ravel_pytree(grad_te)[0]

    # -- test --
    error = jnp.abs(grad_gt - grad_te).mean().item()
    print(error)
    if verbose: print("error: ",error)
    assert error < 1e-5


