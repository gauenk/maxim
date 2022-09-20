
# from .maxim import MAXIM
import ml_collections
from .maxim import Model
from ..common import optional,_MODEL_VARIANT_DICT,_MODEL_CONFIGS
from ..utils.model_utils import get_params

def load_model(*args,**kwargs):
    task = optional(kwargs,'task','denoising')
    ckpt_path = optional(kwargs,'ckpt_path',"weights/denoising/checkpoint.npz")
    model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)
    model_configs.variant = _MODEL_VARIANT_DICT[task]
    model = Model(**model_configs)
    params = get_params(ckpt_path)
    # return MAXIM()
    return model,params


