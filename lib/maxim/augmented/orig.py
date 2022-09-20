# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for the MAXIM model."""

import functools
from typing import Any, Sequence, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

# import dnls.jax as dnls

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




