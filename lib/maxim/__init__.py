
# -- api --
from . import utils

# -- model version --
from .original import load_model as load_original_model
from .augmented import load_model as load_augmented_model


def load_model(*args,**kwargs):
    attn_mode = optional(kwargs,"attn_mode","original")
    if "original" in attn_mode:
        return load_original_model(*args,**kwargs)
    else:
        return load_augmented_model(*args,**kwargs)
