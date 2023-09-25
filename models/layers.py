# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
"""Common layers for defining score networks.
"""
import functools
import math
import string
from typing import Any, Sequence, Optional

import numpy as np
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp


def get_act(config):
  """Get activation functions from the config file."""

  if config.nonlinearity.lower() == 'elu':
    return nn.elu
  elif config.nonlinearity.lower() == 'relu':
    return nn.relu
  elif config.nonlinearity.lower() == 'lrelu':
    return functools.partial(nn.leaky_relu, negative_slope=0.2)
  elif config.nonlinearity.lower() == 'swish':
    return nn.swish
  elif config.nonlinearity.lower() == 'tanh':
    return jnp.tanh
  else:
    raise NotImplementedError('activation function does not exist!')

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jnp.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

###################################################################################################################################################
# code from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html#What-is-Attention?
###################################################################################################################################################

class MultiheadAttention(nn.Module):
  embed_dim : int  # Output dimension
  num_heads : int  # Number of parallel heads (h)

  def setup(self):
    # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
    # Note that in many implementations you see "bias=False" which is optional
    self.qkv_proj = nn.Dense(3*self.embed_dim,
                              kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                              bias_init=nn.initializers.zeros  # Bias init with zeros
                            )
    self.o_proj = nn.Dense(self.embed_dim,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

  def __call__(self, x, mask=None):
    batch_size, seq_length, embed_dim = x.shape
    if mask is not None:
      mask = expand_mask(mask)
    qkv = self.qkv_proj(x)

    # Separate Q, K, V from linear output
    qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
    qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
    q, k, v = jnp.array_split(qkv, 3, axis=-1)

    # Determine value outputs
    values, attention = scaled_dot_product(q, k, v, mask=mask)
    values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
    values = values.reshape(batch_size, seq_length, embed_dim)
    o = self.o_proj(values)

    return o, attention

class PositionalEncoding(nn.Module):
  d_model : int         # Hidden dimensionality of the input.
  max_len : int = 5000  # Maximum length of a sequence to expect.

  def setup(self):
    # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
    pe = np.zeros((self.max_len, self.d_model))
    position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
    div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[None]
    self.pe = jax.device_put(pe)

  def __call__(self, x):
    x = x + self.pe[:, :x.shape[1]]
    return x

def scaled_dot_product(q, k, v, mask=None):
  d_k = q.shape[-1]
  attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
  attn_logits = attn_logits / math.sqrt(d_k)
  if mask is not None:
    attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
  attention = nn.softmax(attn_logits, axis=-1)
  values = jnp.matmul(attention, v)
  return values, attention

def expand_mask(mask):
  assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
  if mask.ndim == 3:
    mask = mask.unsqueeze(1)
  while mask.ndim < 4:
    mask = mask.unsqueeze(0)
  return mask
