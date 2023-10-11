from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np
from typing import Any

from models import utils as mutils


@flax.struct.dataclass
class SamplerState:
  u0: jnp.ndarray

@flax.struct.dataclass
class TimeSampler:
  sample_t: Any


def get_time_sampler(config):

  def sample_uniformly(bs, state, t_0=0.0, t_1=1.0):
    u = (state.u0 + math.sqrt(2)*jnp.arange(bs*jax.device_count())) % 1
    new_state = state.replace(u0=u[-1:])
    t = (t_1-t_0)*u[jax.process_index()*bs:(jax.process_index()+1)*bs] + t_0
    return t, new_state

  sampler = TimeSampler(sample_t=sample_uniformly)
  init_state = SamplerState(u0=jnp.array([0.5]))

  return sampler, init_state


def get_optimizer(config):
  schedule = optax.join_schedules([optax.linear_schedule(0.0, config.lr, config.warmup), 
                                   optax.constant_schedule(config.lr)], 
                                   boundaries=[config.warmup])
  if config.name == 'adam':
    optimizer = optax.adam(learning_rate=schedule, b1=config.beta1, eps=config.eps)
  elif config.name == 'adamw':
    optimizer = optax.adamw(learning_rate=schedule, b1=config.beta1, eps=config.eps)
  elif config.name == 'sgd':
    optimizer = optax.sgd(learning_rate=schedule, momentum=config.beta1)
  else:
    NotImplementedError(f'optimizer {config.optimizer} is not implemented')
  optimizer = optax.chain(
    optax.clip(config.grad_clip),
    optimizer
  )
  return optimizer


def get_step_fn(config, optimizer_s, optimizer_q, loss_fn):

  def step_fn(carry_state, batch):
    (key, state_s, state_q) = carry_state
    key, step_key = jax.random.split(key)
    grad_fn = jax.value_and_grad(loss_fn, argnums=[1,2], has_aux=True)
    (loss, (new_sampler_state, metrics)), grads = grad_fn(step_key, 
      state_s.model_params,
      state_q.model_params,
      state_s.sampler_state, 
      batch)
    
    def update(optimizer, grad, state, every=1):
      updates, opt_state = optimizer.update(grad, state.opt_state, state.model_params)
      new_params = optax.apply_updates(state.model_params, updates)
      new_params_ema = jax.tree_map(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_params
      )
      new_state = state.replace(
        step=state.step+1,
        opt_state=opt_state,
        sampler_state=new_sampler_state, 
        model_params=new_params,
        params_ema=new_params_ema
      )
      return new_state
      
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.tree_map(lambda _metric: jax.lax.pmean(_metric, axis_name='batch'), metrics)
    
    new_state_s = update(optimizer_s, grads[0], state_s)
    new_state_q = update(optimizer_q, grads[1], state_q)
    new_carry_state = (key, new_state_s, new_state_q)
    return new_carry_state, (loss, metrics)

  return step_fn
