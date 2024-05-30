import jax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import functools

from . import utils, layers

get_act = layers.get_act

@utils.register_model(name='mlp_s')
class MLP(nn.Module):
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, t: jnp.ndarray, x: jnp.ndarray, train: bool):
    config = self.config
    act = get_act(config)
    nf = config.nf

    if config.embed_time:
      # temb = layers.get_timestep_embedding(t.ravel(), nf)
      temb = t
      temb = nn.Dense(nf)(temb)
      temb = nn.Dense(nf)(act(temb))
      h = x
    else:
      h = jnp.hstack([x, t])
    
    h = act(nn.Dense(nf)(h))
    for _ in range(config.n_layers):
      if config.embed_time:
        h += temb
      h = nn.Dropout(config.dropout)(h, deterministic=not train)
      if config.skip:
        h = act(nn.Dense(nf)(h)) + h
      else:
        h = act(nn.Dense(nf)(h))
    h = act(nn.Dense(nf)(h)) 
    h = nn.Dense(config.input_dim)(h)
    return (h*x).sum(1, keepdims=True)
  
  
@utils.register_model(name='mlp_scalar_s')
class MLP(nn.Module):
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, t: jnp.ndarray, x: jnp.ndarray, train: bool):
    config = self.config
    act = get_act(config)
    nf = config.nf

    if config.embed_time:
      # temb = layers.get_timestep_embedding(t.ravel(), nf)
      temb = t
      temb = nn.Dense(nf)(temb)
      temb = nn.Dense(nf)(act(temb))
      h = x
    else:
      h = jnp.hstack([x, t])
    
    h = act(nn.Dense(nf)(h))
    for _ in range(config.n_layers):
      if config.embed_time:
        h += temb
      h = nn.Dropout(config.dropout)(h, deterministic=not train)
      if config.skip:
        h = act(nn.Dense(nf)(h)) + h
      else:
        h = act(nn.Dense(nf)(h))
    h = act(nn.Dense(nf)(h)) 
    h = nn.Dense(nf, use_bias=True)(h)
    return h.sum(1, keepdims=True)

  
@utils.register_model(name='mlp_vf')
class MLP(nn.Module):
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, t: jnp.ndarray, x: jnp.ndarray, train: bool):
    config = self.config
    act = get_act(config)
    nf = config.nf

    if config.embed_time:
      # temb = layers.get_timestep_embedding(t.ravel(), nf)
      temb = t
      temb = nn.Dense(nf)(temb)
      temb = nn.Dense(nf)(act(temb))
      h = x
    else:
      h = jnp.hstack([x, t])
    
    h = act(nn.Dense(nf)(h))
    for _ in range(config.n_layers):
      if config.embed_time:
        h += temb
      h = nn.Dropout(config.dropout)(h, deterministic=not train)
      if config.skip:
        h = act(nn.Dense(nf)(h)) + h
      else:
        h = act(nn.Dense(nf)(h))
    h = act(nn.Dense(nf)(h)) 
    h = nn.Dense(config.input_dim)(h)
    return h

  
@utils.register_model(name='mlp_q')
class MLP(nn.Module):
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, t: jnp.ndarray, batch: jnp.ndarray, train: bool):
    config = self.config
    act = get_act(config)
    nf = config.nf
    
    timesteps, x = batch
    # timesteps.shape = (batch_size, n_marginals, 1)
    # x.shape = (batch_size, n_marginals, dim)
    t_right = (timesteps < t.reshape(-1,1,1)).sum(1)
    t_right = jnp.fmax(t_right, jnp.ones_like(t_right).astype(int))
    t_left = t_right - 1

    x_left = x[jnp.arange(len(x)), t_left.ravel(), :]
    x_right = x[jnp.arange(len(x)), t_right.ravel(), :]
    t_0 = timesteps[jnp.arange(len(x)), t_left.ravel(), :]
    t_1 = timesteps[jnp.arange(len(x)), t_right.ravel(), :]
    
    temb = layers.get_timestep_embedding(t.ravel(), nf)
    temb = nn.Dense(nf)(temb)
    temb = nn.Dense(nf)(act(temb))
    
    h = jnp.concatenate([x_left, x_right], 1)
    h = jnp.hstack([t, h])
    # h = jnp.hstack([nn.one_hot(t_left.ravel(), config.n_marginals-1), h])
    if config.indicator:
      h = jnp.hstack([h, t < 0.5*(t_0 + t_1)])
    h = act(nn.Dense(nf)(h))
    
    for _ in range(config.n_layers):
      # h += temb
      h = nn.Dropout(config.dropout)(h, deterministic=not train)
      if config.skip:
        h = act(nn.Dense(nf)(h)) + h
      else:
        h = act(nn.Dense(nf)(h))
    h = nn.Dense(config.input_dim)(h)
    
    out = (t_1-t)/(t_1-t_0)*x_left + (t-t_0)/(t_1-t_0)*x_right
    out += (1.0 - ((t-t_0)/(t_1-t_0))**2 - ((t_1-t)/(t_1-t_0))**2)*h
    return out
  

@utils.register_model(name='attn_q')
class MLP(nn.Module):
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, t: jnp.ndarray, batch: jnp.ndarray, train: bool):
    config = self.config
    act = get_act(config)
    nf = config.nf
    
    timesteps, x = batch
    # timesteps.shape = (batch_size, n_marginals, 1)
    # x.shape = (batch_size, n_marginals, dim)
    t_right = (timesteps < t.reshape(-1,1,1)).sum(1)
    t_right = jnp.fmax(t_right, jnp.ones_like(t_right).astype(int))
    t_left = t_right - 1

    x_left = x[jnp.arange(len(x)), t_left.ravel(), :]
    x_right = x[jnp.arange(len(x)), t_right.ravel(), :]
    t_0 = timesteps[jnp.arange(len(x)), t_left.ravel(), :]
    t_1 = timesteps[jnp.arange(len(x)), t_right.ravel(), :]
    
    h = jnp.concatenate([x, jnp.ones((x.shape[0], x.shape[1], 1))*t[:,None,:]], axis=-1)
    # if config.indicator:
    #   h = jnp.hstack([h, jnp.ones((x.shape[0], x.shape[1], 1))*t[:,None,:] < 0.5*(t_0 + t_1)])
    h = nn.Dense(nf)(h)
    h = layers.PositionalEncoding(nf)(h)
    for _ in range(config.n_layers):
      attn_out, _ = layers.MultiheadAttention(nf, 4)(h)
      h = h + nn.Dropout(config.dropout)(attn_out, deterministic=not train)
      h = nn.LayerNorm()(h)
    h = h.sum(1)
    h = nn.Dense(config.input_dim)(h)
    
    out = (t_1-t)/(t_1-t_0)*x_left + (t-t_0)/(t_1-t_0)*x_right
    out += (1.0 - ((t-t_0)/(t_1-t_0))**2 - ((t_1-t)/(t_1-t_0))**2)*h
    return out
