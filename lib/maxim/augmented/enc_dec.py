# -- imports --
import functools
from typing import Any, Sequence, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

# -- local --
from .gating import CrossGatingBlock
from .attn import ResidualSplitHeadMultiAxisGmlpLayer,RCAB
from .helpers import block_images_einops,unblock_images_einops

# -- conv layers --
Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1))
ConvT_up = functools.partial(nn.ConvTranspose,
                             kernel_size=(2, 2),
                             strides=(2, 2))
Conv_down = functools.partial(nn.Conv,
                              kernel_size=(4, 4),
                              strides=(2, 2))


class UNetEncoderBlock(nn.Module):
  """Encoder block in MAXIM."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  num_groups: int = 1
  lrelu_slope: float = 0.2
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  channels_reduction: int = 4
  dropout_rate: float = 0.0
  downsample: bool = True
  use_global_mlp: bool = True
  use_bias: bool = True
  use_cross_gating: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, skip: jnp.ndarray = None,
               enc: jnp.ndarray = None, dec: jnp.ndarray = None, *,
               deterministic: bool = True) -> jnp.ndarray:
    if skip is not None:
      x = jnp.concatenate([x, skip], axis=-1)

    # convolution-in
    x = Conv1x1(self.features, use_bias=self.use_bias)(x)
    shortcut_long = x

    for i in range(self.num_groups):
      if self.use_global_mlp:
        x = ResidualSplitHeadMultiAxisGmlpLayer(
            grid_size=self.grid_size,
            block_size=self.block_size,
            grid_gmlp_factor=self.grid_gmlp_factor,
            block_gmlp_factor=self.block_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            name=f"SplitHeadMultiAxisGmlpLayer_{i}")(x, deterministic)
      x = RCAB(
          features=self.features,
          reduction=self.channels_reduction,
          use_bias=self.use_bias,
          name=f"channel_attention_block_1{i}")(x)

    x = x + shortcut_long

    if enc is not None and dec is not None:
      assert self.use_cross_gating
      x, _ = CrossGatingBlock(
          features=self.features,
          block_size=self.block_size,
          grid_size=self.grid_size,
          dropout_rate=self.dropout_rate,
          input_proj_factor=self.input_proj_factor,
          upsample_y=False,
          use_bias=self.use_bias,
          name="cross_gating_block")(
              x, enc + dec, deterministic=deterministic)

    if self.downsample:
      x_down = Conv_down(self.features, use_bias=self.use_bias)(x)
      return x_down, x
    else:
      return x


class UNetDecoderBlock(nn.Module):
  """Decoder block in MAXIM."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  num_groups: int = 1
  lrelu_slope: float = 0.2
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  channels_reduction: int = 4
  dropout_rate: float = 0.0
  downsample: bool = True
  use_global_mlp: bool = True
  use_bias: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, bridge: jnp.ndarray = None,
               deterministic: bool = True) -> jnp.ndarray:
    x = ConvT_up(self.features, use_bias=self.use_bias)(x)

    x = UNetEncoderBlock(
        self.features,
        num_groups=self.num_groups,
        lrelu_slope=self.lrelu_slope,
        block_size=self.block_size,
        grid_size=self.grid_size,
        block_gmlp_factor=self.block_gmlp_factor,
        grid_gmlp_factor=self.grid_gmlp_factor,
        channels_reduction=self.channels_reduction,
        use_global_mlp=self.use_global_mlp,
        dropout_rate=self.dropout_rate,
        downsample=False,
        use_bias=self.use_bias)(x, skip=bridge, deterministic=deterministic)
    return x


