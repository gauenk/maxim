
# -- imports --
import functools
from typing import Any, Sequence, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

# import dnls.jax as dnls

# -- local --
from .gating import GridGatingUnit,BlockGatingUnit
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
weight_initializer = nn.initializers.normal(stddev=2e-2)


class BottleneckBlock(nn.Module):
  """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  num_groups: int = 1
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  channels_reduction: int = 4
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic):
    """Applies the Mixer block to inputs."""
    assert x.ndim == 4  # Input has shape [batch, h, w, c]
    n, h, w, num_channels = x.shape

    # input projection
    x = Conv1x1(self.features, use_bias=self.use_bias, name="input_proj")(x)
    shortcut_long = x

    for i in range(self.num_groups):
      x = ResidualSplitHeadMultiAxisGmlpLayer(
          grid_size=self.grid_size,
          block_size=self.block_size,
          grid_gmlp_factor=self.grid_gmlp_factor,
          block_gmlp_factor=self.block_gmlp_factor,
          input_proj_factor=self.input_proj_factor,
          use_bias=self.use_bias,
          dropout_rate=self.dropout_rate,
          name=f"SplitHeadMultiAxisGmlpLayer_{i}")(x, deterministic)
      # Channel-mixing part, which provides within-patch communication.
      x = RDCAB(
          features=self.features,
          reduction=self.channels_reduction,
          use_bias=self.use_bias,
          name=f"channel_attention_block_1_{i}")(
              x)

    # long skip-connect
    x = x + shortcut_long
    return x


class MlpBlock(nn.Module):
  """A 1-hidden-layer MLP block, applied over the last dimension."""
  mlp_dim: int
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, d = x.shape
    x = nn.Dense(self.mlp_dim, use_bias=self.use_bias,
                 kernel_init=weight_initializer)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn.Dense(d, use_bias=self.use_bias,
                 kernel_init=weight_initializer)(x)
    return x


class UpSampleRatio(nn.Module):
  """Upsample features given a ratio > 0."""
  features: int
  ratio: float
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    n, h, w, c = x.shape
    x = jax.image.resize(
        x,
        shape=(n, int(h * self.ratio), int(w * self.ratio), c),
        method="bilinear")
    x = Conv1x1(features=self.features, use_bias=self.use_bias)(x)
    return x


class CALayer(nn.Module):
  """Squeeze-and-excitation block for channel attention.

  ref: https://arxiv.org/abs/1709.01507
  """
  features: int
  reduction: int = 4
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    # 2D global average pooling
    y = jnp.mean(x, axis=[1, 2], keepdims=True)
    # Squeeze (in Squeeze-Excitation)
    y = Conv1x1(self.features // self.reduction, use_bias=self.use_bias)(y)
    y = nn.relu(y)
    # Excitation (in Squeeze-Excitation)
    y = Conv1x1(self.features, use_bias=self.use_bias)(y)
    y = nn.sigmoid(y)
    return x * y


class RCAB(nn.Module):
  """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""
  features: int
  reduction: int = 4
  lrelu_slope: float = 0.2
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    shortcut = x
    x = nn.LayerNorm(name="LayerNorm")(x)
    x = Conv3x3(features=self.features, use_bias=self.use_bias, name="conv1")(x)
    x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
    x = Conv3x3(features=self.features, use_bias=self.use_bias, name="conv2")(x)
    x = CALayer(features=self.features, reduction=self.reduction,
                use_bias=self.use_bias, name="channel_attention")(x)
    return x + shortcut

class GridGmlpLayer(nn.Module):
  """Grid gMLP layer that performs global mixing of tokens."""
  grid_size: Sequence[int]
  use_bias: bool = True
  factor: int = 2
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, num_channels = x.shape
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    x = block_images_einops(x, patch_size=(fh, fw))
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    y = nn.LayerNorm(name="LayerNorm")(x)
    y = nn.Dense(num_channels * self.factor, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="in_project")(y)
    y = nn.gelu(y)
    y = GridGatingUnit(use_bias=self.use_bias, name="GridGatingUnit")(y)
    y = nn.Dense(num_channels, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="out_project")(y)
    y = nn.Dropout(self.dropout_rate)(y, deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    return x

class BlockGmlpLayer(nn.Module):
  """Block gMLP layer that performs local mixing of tokens."""
  block_size: Sequence[int]
  use_bias: bool = True
  factor: int = 2
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, num_channels = x.shape
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw
    x = block_images_einops(x, patch_size=(fh, fw))
    # MLP2: Local (block) mixing part, provides within-block communication.
    y = nn.LayerNorm(name="LayerNorm")(x)
    y = nn.Dense(num_channels * self.factor, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="in_project")(y)
    y = nn.gelu(y)
    y = BlockGatingUnit(use_bias=self.use_bias, name="BlockGatingUnit")(y)
    y = nn.Dense(num_channels, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="out_project")(y)
    y = nn.Dropout(self.dropout_rate)(y, deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
  """The multi-axis gated MLP block."""
  block_size: Sequence[int]
  grid_size: Sequence[int]
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  use_bias: bool = True
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    shortcut = x
    n, h, w, num_channels = x.shape
    x = nn.LayerNorm(name="LayerNorm_in")(x)
    x = nn.Dense(num_channels * self.input_proj_factor, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="in_project")(x)
    x = nn.gelu(x)

    u, v = jnp.split(x, 2, axis=-1)
    # GridGMLPLayer
    u = GridGmlpLayer(
        grid_size=self.grid_size,
        factor=self.grid_gmlp_factor,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        name="GridGmlpLayer")(u, deterministic)

    # BlockGMLPLayer
    v = BlockGmlpLayer(
        block_size=self.block_size,
        factor=self.block_gmlp_factor,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        name="BlockGmlpLayer")(v, deterministic)

    x = jnp.concatenate([u, v], axis=-1)

    x = nn.Dense(num_channels, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="out_project")(x)
    x = nn.Dropout(self.dropout_rate)(x, deterministic)
    x = x + shortcut
    return x


class RDCAB(nn.Module):
  """Residual dense channel attention block. Used in Bottlenecks."""
  features: int
  reduction: int = 16
  use_bias: bool = True
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    y = nn.LayerNorm(name="LayerNorm")(x)
    y = MlpBlock(
        mlp_dim=self.features,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        name="channel_mixing")(
            y, deterministic=deterministic)
    y = CALayer(
        features=self.features,
        reduction=self.reduction,
        use_bias=self.use_bias,
        name="channel_attention")(
            y)
    x = x + y
    return x

class SAM(nn.Module):
  """Supervised attention module for multi-stage training.

  Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
  """
  features: int
  output_channels: int = 3
  use_bias: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, x_image: jnp.ndarray, *,
               train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the SAM module to the input and features.

    Args:
      x: the output features from UNet decoder with shape (h, w, c)
      x_image: the input image with shape (h, w, 3)
      train: Whether it is training

    Returns:
      A tuple of tensors (x1, image) where (x1) is the sam features used for the
        next stage, and (image) is the output restored image at current stage.
    """
    # Get features
    x1 = Conv3x3(self.features, use_bias=self.use_bias)(x)

    # Output restored image X_s
    if self.output_channels == 3:
      image = Conv3x3(self.output_channels, use_bias=self.use_bias)(x) + x_image
    else:
      image = Conv3x3(self.output_channels, use_bias=self.use_bias)(x)

    # Get attention maps for features
    x2 = nn.sigmoid(Conv3x3(self.features, use_bias=self.use_bias)(image))

    # Get attended feature maps
    x1 = x1 * x2

    # Residual connection
    x1 = x1 + x
    return x1, image


