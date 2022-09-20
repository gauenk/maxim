# -- imports --
import functools
from typing import Any, Sequence, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

# -- local --
from .helpers import block_images_einops,unblock_images_einops

# -- conv --
Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1))
ConvT_up = functools.partial(nn.ConvTranspose,
                             kernel_size=(2, 2),
                             strides=(2, 2))
Conv_down = functools.partial(nn.Conv,
                              kernel_size=(4, 4),
                              strides=(2, 2))
weight_initializer = nn.initializers.normal(stddev=2e-2)



class BlockGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.

  The 'spatial' dim is defined as the **second last**.
  If applied on other dims, you should swapaxes first.
  """
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    u, v = jnp.split(x, 2, axis=-1)
    v = nn.LayerNorm(name="intermediate_layernorm")(v)
    n = x.shape[-2]  # get spatial dim
    v = jnp.swapaxes(v, -1, -2)
    v = nn.Dense(n, use_bias=self.use_bias, kernel_init=weight_initializer)(v)
    v = jnp.swapaxes(v, -1, -2)
    return u * (v + 1.)

class GridGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.

  The 'spatial' dim is defined as the second last.
  If applied on other dims, you should swapaxes first.
  """
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    u, v = jnp.split(x, 2, axis=-1)
    v = nn.LayerNorm(name="intermediate_layernorm")(v)
    n = x.shape[-3]   # get spatial dim
    v = jnp.swapaxes(v, -1, -3)
    v = nn.Dense(n, use_bias=self.use_bias, kernel_init=weight_initializer)(v)
    v = jnp.swapaxes(v, -1, -3)
    return u * (v + 1.)


class GetSpatialGatingWeights(nn.Module):
  """Get gating weights for cross-gating MLP block."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  input_proj_factor: int = 2
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic):
    n, h, w, num_channels = x.shape

    # input projection
    x = nn.LayerNorm(name="LayerNorm_in")(x)
    x = nn.Dense(
        num_channels * self.input_proj_factor,
        use_bias=self.use_bias,
        name="in_project")(
            x)
    x = nn.gelu(x)
    u, v = jnp.split(x, 2, axis=-1)

    # Get grid MLP weights
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    u = block_images_einops(u, patch_size=(fh, fw))
    dim_u = u.shape[-3]
    u = jnp.swapaxes(u, -1, -3)
    u = nn.Dense(
        dim_u, use_bias=self.use_bias, kernel_init=nn.initializers.normal(2e-2),
        bias_init=nn.initializers.ones)(u)
    u = jnp.swapaxes(u, -1, -3)
    u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))

    # Get Block MLP weights
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw
    v = block_images_einops(v, patch_size=(fh, fw))
    dim_v = v.shape[-2]
    v = jnp.swapaxes(v, -1, -2)
    v = nn.Dense(
        dim_v, use_bias=self.use_bias, kernel_init=nn.initializers.normal(2e-2),
        bias_init=nn.initializers.ones)(v)
    v = jnp.swapaxes(v, -1, -2)
    v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))

    x = jnp.concatenate([u, v], axis=-1)
    x = nn.Dense(num_channels, use_bias=self.use_bias, name="out_project")(x)
    x = nn.Dropout(self.dropout_rate)(x, deterministic)
    return x


class CrossGatingBlock(nn.Module):
  """Cross-gating MLP block."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  dropout_rate: float = 0.0
  input_proj_factor: int = 2
  upsample_y: bool = True
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, y, deterministic=True):
    # Upscale Y signal, y is the gating signal.
    if self.upsample_y:
      y = ConvT_up(self.features, use_bias=self.use_bias)(y)

    x = Conv1x1(self.features, use_bias=self.use_bias)(x)
    n, h, w, num_channels = x.shape
    y = Conv1x1(num_channels, use_bias=self.use_bias)(y)

    assert y.shape == x.shape
    shortcut_x = x
    shortcut_y = y

    # Get gating weights from X
    x = nn.LayerNorm(name="LayerNorm_x")(x)
    x = nn.Dense(num_channels, use_bias=self.use_bias, name="in_project_x")(x)
    x = nn.gelu(x)
    gx = GetSpatialGatingWeights(
        features=num_channels,
        block_size=self.block_size,
        grid_size=self.grid_size,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        name="SplitHeadMultiAxisGating_x")(
            x, deterministic=deterministic)

    # Get gating weights from Y
    y = nn.LayerNorm(name="LayerNorm_y")(y)
    y = nn.Dense(num_channels, use_bias=self.use_bias, name="in_project_y")(y)
    y = nn.gelu(y)
    gy = GetSpatialGatingWeights(
        features=num_channels,
        block_size=self.block_size,
        grid_size=self.grid_size,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        name="SplitHeadMultiAxisGating_y")(
            y, deterministic=deterministic)

    # Apply cross gating: X = X * GY, Y = Y * GX
    y = y * gx
    y = nn.Dense(num_channels, use_bias=self.use_bias, name="out_project_y")(y)
    y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
    y = y + shortcut_y

    x = x * gy  # gating x using y
    x = nn.Dense(num_channels, use_bias=self.use_bias, name="out_project_x")(x)
    x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
    x = x + y + shortcut_x  # get all aggregated signals
    return x, y
