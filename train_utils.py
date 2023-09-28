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


# def get_intermediate_generator(model, config, dynamics):

#   def generate_intermediate(key, state, batch):
#     s = mutils.get_model_fn(model, state.model_params, train=False)
#     dsdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=1)
#     data = batch['image'][0]
#     keys = jax.random.split(key)
#     t_0 = jnp.zeros((data.shape[0],1,1,1))
#     x_0, _, _ = dynamics(keys[0], data, t_0)
#     dt = 1e-2
#     num_ode_steps = int(1.0/dt)
#     def ode_step(carry_state, key):
#       x, t = carry_state
#       next_x = x + dt*dsdx_fn(t, x, key)
#       next_t = t + dt
#       return (next_x, next_t), next_x
#     x_t = jax.lax.scan(ode_step, (x_0, t_0), jax.random.split(keys[1], num_ode_steps))[1]
#     x_t = jnp.vstack([jnp.expand_dims(x_0, 0), x_t])
#     x_t = jax.lax.stop_gradient(x_t)
#     return x_t

#   return generate_intermediate


# def get_x_t_sampler(config, dynamics):

#   def schedule(step):
#     return jnp.min(jnp.array([step/20000, 1.0]))
  
#   def sample_x_t(key, batch, ode_x_t, step):
#     data = batch['image'][:1]
#     keys = jax.random.split(key)
#     t = jnp.linspace(0.0,1.0,101).reshape((101,1,1,1,1))
#     _, _, x_t = dynamics(keys[0], data, t)
#     mask = jax.random.uniform(keys[1], (x_t.shape[0],x_t.shape[1],1,1,1)) < schedule(step)
#     mask = mask.astype(float)
#     x_t = mask*ode_x_t + (1-mask)*x_t
#     return x_t

#   return sample_x_t

# def get_artifact_generator(model, config, artifact_shape):
#   if 'am' == config.loss:
#     generator = get_ot_generator(model, config, artifact_shape)
#   elif 'rf' == config.loss:  
#     generator = get_ot_generator(model, config, artifact_shape)
#   else:
#     raise NotImplementedError(f'generator for {config.model.loss} is not implemented')
#   return generator


# def get_ode_generator(model, config, dynamics, artifact_shape):

#   def artifact_generator(key, state, batch):
#     x_0, _, _ = dynamics(key, batch, t=jnp.zeros((1)))
    
#     def vector_field(t,y,state):
#       s = mutils.get_model_fn(model, 
#                               state.params_ema if config.eval.use_ema else state.model_params, 
#                               train=False)
#       dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
#       return dsdx(t,y)
#     solve = partial(diffrax.diffeqsolve, 
#                     terms=diffrax.ODETerm(vector_field), 
#                     solver=diffrax.Euler(), 
#                     t0=0.0, t1=1.0, dt0=1e-2, 
#                     saveat=diffrax.SaveAt(ts=[1.0]),
#                     stepsize_controller=diffrax.ConstantStepSize(True), 
#                     adjoint=diffrax.NoAdjoint())
  
#     solution = solve(y0=x_0, args=state)
#     return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']
    
#   return artifact_generator


# def get_sde_generator(model, config, dynamics, artifact_shape):

#   def artifact_generator(key, state, batch):
#     x_0, _, _ = dynamics(key, batch, t=jnp.zeros((1)))

#     def vector_field(t,y,state):
#       s = mutils.get_model_fn(model, 
#                               state.params_ema if config.eval.use_ema else state.model_params, 
#                               train=False)
#       dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
#       return dsdx(t,y)
    
#     diffusion = lambda t, y, args: config.model.sigma * jnp.ones(x_0.shape)
#     brownian_motion = diffrax.UnsafeBrownianPath(shape=x_0.shape, key=key)
#     terms = diffrax.MultiTerm(diffrax.ODETerm(vector_field), 
#                               diffrax.WeaklyDiagonalControlTerm(diffusion, brownian_motion))
#     solve = partial(diffrax.diffeqsolve, 
#                     terms=terms, 
#                     solver=diffrax.Euler(), 
#                     t0=0.0, t1=1.0, dt0=1e-2, 
#                     saveat=diffrax.SaveAt(ts=[1.0]),
#                     stepsize_controller=diffrax.ConstantStepSize(True), 
#                     adjoint=diffrax.NoAdjoint())

#     solution = solve(y0=x_0, args=state)
#     return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']

#   return artifact_generator


# def get_ot_generator(model, config, artifact_shape):

#   def artifact_generator(key, state, x_0):
#     x_0 = x_0[:x_0.shape[0]//2]
#     s = mutils.get_model_fn(model, 
#                             state.params_ema if config.eval.use_ema else state.model_params, 
#                             train=False)
#     if 'unet' == config.model_s.name:
#       dsdx = s
#     else:
#       dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
#     vector_field = lambda _t,_x,_args: dsdx(_t,_x)
#     solve = partial(diffrax.diffeqsolve, 
#                     terms=diffrax.ODETerm(vector_field), 
#                     solver=diffrax.Euler(), 
#                     t0=0.0, t1=1.0, dt0=1e-2, 
#                     saveat=diffrax.SaveAt(ts=[1.0]),
#                     stepsize_controller=diffrax.ConstantStepSize(True), 
#                     adjoint=diffrax.NoAdjoint())
#     solution = solve(y0=x_0, args=state)
#     one_step_artifacts = x_0 + dsdx(jnp.zeros((x_0.shape[0], 1, 1, 1)), x_0)
#     artifacts = jnp.stack([solution.ys[-1][:,:,:,:artifact_shape[3]], one_step_artifacts], 0)
#     return artifacts, solution.stats['num_steps']
    
#   return artifact_generator
