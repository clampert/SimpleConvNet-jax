#!/usr/bin/env python3
# coding: utf-8

from functools import partial 
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any

class ConvBlock(nn.Module):
  """ResNet block."""
  channels_out: int
  use_bias: bool = False
  use_maxpool: bool = True
  dtype: str = jnp.float32
  
  @nn.compact
  def __call__(self, x, train: bool = False):
    x = nn.Conv(self.channels_out, kernel_size=(3,3), padding=(1,1), use_bias=self.use_bias, dtype=self.dtype)(x)
    if self.use_maxpool:
      x = nn.max_pool(x, window_shape=(2,2), strides=(2,2)) # in pytorch, stride is implicit. Here, we have to set it!
    x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(self.channels_out, kernel_size=(3,3), padding=(1,1), use_bias=self.use_bias, dtype=self.dtype)(x)
    x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype)(x)
    x = nn.relu(x)
    return x


scaler = 2.
base = 64
num_channels = {
  'block0': round(scaler**0 * base), # 64  w/ scaler 2 at base 64
  'block1': round(scaler**1 * base), # 128 w/ scaler 2 at base 64
  'block2': round(scaler**2 * base), # 256 w/ scaler 2 at base 64
  'block3': round(scaler**3 * base), # 512 w/ scaler 2 at base 64
  'num_classes': 10
}

class SpeedyResNet(nn.Module):
  dtype: Any = jnp.float32
    
  @nn.compact
  def __call__(self, x, train=False):
    x = ConvBlock(num_channels['block0'], use_maxpool=False, dtype=self.dtype)(x, train=train)
    x = ConvBlock(num_channels['block1'], dtype=self.dtype)(x, train=train)
    x = ConvBlock(num_channels['block2'], dtype=self.dtype)(x, train=train)
    x = ConvBlock(num_channels['block3'], dtype=self.dtype)(x, train=train)
    x = jnp.amax(x, axis=(1,2)) # => FastGlobalMaxPooling()(x)
    x = nn.Dense(num_channels['num_classes'], use_bias=False, dtype=self.dtype)(x)
    return x
