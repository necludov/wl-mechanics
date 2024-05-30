from typing import Any
from functools import partial

import math
import numpy as np

import jax
import jax.numpy as jnp
import flax
import diffrax
import ot

from models import utils as mutils


def get_generator(model, config):
  
  def grad_vf(t,y,state):
    s = mutils.get_model_fn(model, 
                            state.params_ema if config.eval.use_ema else state.model_params, 
                            train=False)
    dsdx = jax.grad(lambda _t, _x: s(_t*jnp.ones((_x.shape[0],1)), _x).sum(), argnums=1)
    return dsdx(t,y)
  
  def vf(t,y,state):
    s = mutils.get_model_fn(model, 
                            state.params_ema if config.eval.use_ema else state.model_params, 
                            train=False)
    return s(t*jnp.ones((y.shape[0],1)), y)
  
  if config.loss == 'rf':
    vector_field = vf
  else:
    vector_field = grad_vf

  def ode_generator(key, state, batch):
    x_0, t_0, x_1, t_1 = batch
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Dopri5(), 
                    t0=t_0, t1=t_1, dt0=1e-3, 
                    saveat=diffrax.SaveAt(ts=[t_1]),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5))
  
    solution = solve(y0=x_0, args=state)
    return (solution.ys[-1], None), solution.stats['num_steps']
  
  def sde_generator(key, state, batch):
    x_0, t_0, x_1, t_1 = batch
    diffusion = lambda t, y, args: config.sigma * jnp.ones(x_0.shape)
    brownian_motion = diffrax.UnsafeBrownianPath(shape=jax.ShapeDtypeStruct(x_0.shape, jnp.float32), key=key)
    terms = diffrax.MultiTerm(diffrax.ODETerm(vector_field), 
                              diffrax.WeaklyDiagonalControlTerm(diffusion, brownian_motion))
    solve = partial(diffrax.diffeqsolve, 
                    terms=terms, 
                    solver=diffrax.Euler(), 
                    t0=t_0, t1=t_1, dt0=1e-3, 
                    saveat=diffrax.SaveAt(ts=[t_1]),
                    stepsize_controller=diffrax.ConstantStepSize(),
                    adjoint=diffrax.DirectAdjoint())

    solution = solve(y0=x_0, args=state)
    return (solution.ys[-1], None), solution.stats['num_steps']
  
  def ub_generator(key, state, batch):
    def grad_vf_weighted(t,y,state):
      y, w = y
      s = mutils.get_model_fn(model, 
                              state.params_ema if config.eval.use_ema else state.model_params, 
                              train=False)
      dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
      s_val = s(t*jnp.ones((y.shape[0],1)), y)
      return (dsdx(t*jnp.ones((y.shape[0],1)),y), config.lambd*s_val)
    
    x_0, t_0, x_1, t_1 = batch
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(grad_vf_weighted), 
                    solver=diffrax.Dopri5(), 
                    t0=t_0, t1=t_1, dt0=1e-3, 
                    saveat=diffrax.SaveAt(ts=[t_1]),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
                    max_steps=None)

    solution = solve(y0=(x_0, jnp.zeros((x_0.shape[0],1))), args=state)
    x, logw = solution.ys[0][-1], solution.ys[1][-1]
    logw = logw.ravel()
    w = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
    return (x, w), solution.stats['num_steps']
  
  def ot_generator(key, state, batch):
    x_0, t_0, x_1, t_1 = batch
    solution = x_0 + (t_1-t_0)*vector_field(t_0, x_0, state)
    return (solution, None), 1
    
  if config.loss == 'sb':
    return sde_generator, ot_generator
  elif config.loss == 'ubot':
    return ub_generator, ot_generator
  elif config.loss == 'ubot+':
    return ub_generator, ot_generator
  return ode_generator, ot_generator


def get_w1(M, w_x=None, w_y=None):
  def get_w(w, n):
    if w is None:
      w = np.ones(n)
    w = np.array(w).astype(np.float64)
    w /= w.sum()
    return w
  M = np.array(M).astype(np.float64)
  w_x, w_y = get_w(w_x, M.shape[0]), get_w(w_y, M.shape[1])
  return ot.emd2(w_x, w_y, M, numItermax=1e7)
  


from sklearn.metrics.pairwise import rbf_kernel

def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))
