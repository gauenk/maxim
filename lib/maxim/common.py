

import numpy as np

_MODEL_CONFIGS = {
    'variant': '',
    'dropout_rate': 0.0,
    'num_outputs': 3,
    'use_bias': True,
    'num_supervision_scales': 3,
}


_MODEL_VARIANT_DICT = {
    'denoising': 'S-3',
    'deblurring': 'S-3',
    'deraining': 'S-2',
    'dehazing': 'S-2',
    'enhancement': 'S-2',
}


def get_params(ckpt_path):
  """Get params checkpoint."""

  with tf.io.gfile.GFile(ckpt_path, 'rb') as f:
    data = f.read()
  values = np.load(io.BytesIO(data))
  params = recover_tree(*zip(*values.items()))
  params = params['opt']['target']

  return params

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict: return pydict[field]
    else: return default

def select_sigma(data_sigma):
    model_sigma_list = np.array([15,25,50])
    idx = np.argmin(np.abs(model_sigma_list - data_sigma))
    model_sigma = model_sigma_list[idx]
    return model_sigma
