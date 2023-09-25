import math

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils


def get_loss(config, model_s, model_q, time_sampler, train):
  if config.loss == 'am':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'phot':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'sb':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'ubot':
    return get_loss_ours(config, model_s, model_q, time_sampler, train)
  elif config.loss == 'rf':
    return get_loss_rf(config, model_s, model_q, time_sampler, train)
  else:
    NotImplementedError(f'config.loss: {config.loss} is not implemented')


def get_loss_ours(config, model_s, model_q, time_sampler, train):
  
  if config.loss == 'am':
    def potential(_t, _x, _key, _s):
      dsdtdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=[0,1])
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True)
  elif config.loss == 'phot':
    physical_potential = get_physical_potential(config)
    def potential(_t, _x, _key, _s):
      dsdtdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=[0,1])
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      center = jnp.array([[1.0, 0.0]])
      return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True) + physical_potential(_t, _x)
  elif config.loss == 'sb':
    def potential(_t, _x, _key, _s):
      keys = random.split(_key, 3)
      dsdt_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=0)
      dsdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=1)
      
      eps = random.randint(keys[0], _x.shape, 0, 2).astype(float)*2 - 1.0
      dsdx_val, jvp_val = jax.jvp(lambda __x: dsdx_fn(_t, __x, keys[1]), (_x,), (eps,))
      dsdt_val = dsdt_fn(_t, _x, keys[1])
      out = dsdt_val + 0.5*(dsdx_val**2).sum(1, keepdims=True)
      out += 0.5*config.sigma**2*(jvp_val*eps).sum(1, keepdims=True)
      return out
    # def potential(_t, _x, _key, _s):
    #   keys = random.split(_key, 3)
    #   dsdt_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=0)
    #   dsdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=1)
      
    #   dsdt_val = dsdt_fn(_t, _x, keys[1])
    #   dsdx_val = dsdx_fn(_t, _x, keys[1])
    #   out = dsdt_val + 0.5*(dsdx_val**2).sum(1, keepdims=True)
    #   jacobian = jax.jacrev(lambda __x: dsdx_fn(_t, __x, keys[1]).sum(0))(_x)
    #   print(jacobian.shape, 'jacobian.shape', flush=True)
    #   laplacian = jnp.trace(jacobian.transpose((1,0,2)), axis1=1, axis2=2)
    #   print(laplacian.shape, 'laplacian.shape', flush=True)
    #   out += 0.5*config.sigma**2*laplacian[:, None]
    #   return out
  elif config.loss == 'ubot':
    def potential(_t, _x, _key, _s):
      dsdtdx_fn = jax.grad(lambda __t, __x, __key: _s(__t, __x, __key).sum(), argnums=[0,1])
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      return dsdt + 0.5*(dsdx**2).sum(1, keepdims=True) + 0.5*(_s(_t, _x, _key)**2)
  else:
    NotImplementedError(f'potential for config.loss: {config.loss} is not implemented')

  def loss_fn(key, params_s, params_q, sampler_state, batch):
    timesteps, x = batch
    bs = x.shape[0]
    
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model_s, params_s, train=train)
    q = mutils.get_model_fn(model_q, params_q, train=train)
    
    ################################################# loss s #################################################
    acceleration_fn = jax.grad(lambda _t, _x, _key: potential(_t, _x, _key, s).sum(), argnums=1)
    
    # sample time
    t_0, t_1 = timesteps[:,0,:], timesteps[:,-1,:]
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = t.reshape(-1,1)

    # sample data
    samples_q = q(t, batch, keys[0])
    x_t = jax.lax.stop_gradient(samples_q)
    mask = (t >= timesteps[:,:-1,0])*(t <= timesteps[:,1:,0])
    t_mult = config.train.step_size*((1.0 - ((t-timesteps[:,:-1,0])/(timesteps[:,1:,0]-timesteps[:,:-1,0]))**2*mask -\
        ((timesteps[:,1:,0]-t)/(timesteps[:,1:,0]-timesteps[:,:-1,0]))**2*mask)*mask).sum(1, keepdims=True)
    for i in range(config.train.n_gradient_steps):
      dx = jax.lax.stop_gradient(acceleration_fn(t, x_t, jax.random.fold_in(keys[1], i)))
      x_t = x_t + t_mult*jnp.clip(dx, -1, 1)
    
    # boundaries loss
    x_0, x_1 = x[:,0,:], x[:,-1,:]
    s_0 = s(t_0, x_0, keys[2])
    s_1 = s(t_1, x_1, keys[3])
    loss_s = s_0.reshape((-1,1)) - s_1.reshape((-1,1))
    print(loss_s.shape, 'boundaries.shape', flush=True)

    # time loss
    potential_value = potential(t, x_t, keys[4], s)
    loss_s += potential_value
    print(loss_s.shape, 'final.shape', flush=True)
    metrics = {}
    metrics['loss_s'] = loss_s.mean()
    total_loss = loss_s.mean()
    
    ################################################# loss q #################################################
    
    s_detached = mutils.get_model_fn(model_s, jax.lax.stop_gradient(params_s), train=train)
    loss_q = -potential(t, samples_q, keys[5], s_detached)
    metrics['loss_q'] = loss_q.mean()
    total_loss += loss_q.mean()
    
    metrics['acceleration'] = jnp.linalg.norm(acceleration_fn(t, samples_q, keys[6]), axis=1).mean()
    potential_value = jax.lax.stop_gradient(potential_value.squeeze())
    metrics['potential_var'] = ((potential_value.mean() - potential_value)**2).mean()
    
    return total_loss, (next_sampler_state, metrics)

  return loss_fn

def get_loss_rf(config, model_s, model_q, time_sampler, train):

  def loss_fn(key, params_s, params_q, sampler_state, batch):
    timesteps, x = batch
    bs = x.shape[0]
    
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model_s, params_s, train=train)
    q = mutils.get_model_fn(model_q, params_q, train=train)
    
    ################################################# loss s #################################################
    dqdt = jax.grad(lambda _t, _x, _key: q(_t, _x, _key).sum(), argnums=0)
    
    # sample time
    # t_0, t_1 = timesteps[:,0,:], timesteps[:,-1,:]
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = t.reshape(-1,1)
    
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
    
    x_t = (t_1-t)/(t_1-t_0)*x_left + (t-t_0)/(t_1-t_0)*x_right
    dxtdt = (x_right - x_left)/(t_1-t_0)
    
    # loss
    v = s(t, x_t, keys[1])
    loss_s = ((v - dxtdt)**2).sum(-1, keepdims=True)
    print(loss_s.shape, 'final.shape', flush=True)
    metrics = {}
    metrics['loss_s'] = loss_s.mean()
    total_loss = loss_s.mean()
    
    ################################################# loss q #################################################
    
    dvdt = jax.jacrev(lambda _t: s(_t, x_t, keys[1]).sum(0))(t)
    print(dvdt.shape, 'dvdt.shape', flush=True)
    dvdt = jnp.squeeze(dvdt).transpose((1,0))
    print(dvdt.shape, 'dvdt.shape', flush=True)
    dvdxv = jax.jvp(lambda _x: s(t, _x, keys[1]), (x_t,), (v,))[0]
    print(dvdxv.shape, 'dvdxv.shape', flush=True)
    acceleration = dvdt + dvdxv
    
    metrics['acceleration'] = jnp.linalg.norm(acceleration, axis=1).mean()
    
    return total_loss, (next_sampler_state, metrics)

  return loss_fn

import datasets
import numpy as np


def get_physical_potential(config):
  init_key = random.PRNGKey(0)
  X, _ = datasets.get_data(config, init_key)
  t = np.linspace(0.0, 1.0, len(X)).tolist()
  
  t_grid = []
  acc_grid = []
  for i in range(1, len(X)-1):
    v_prev = (X[i].mean(0) - X[i-1].mean(0))/(t[i]-t[i-1])
    v_next = (X[i+1].mean(0) - X[i].mean(0))/(t[i+1]-t[i])
    acc_grid.append((v_next - v_prev)/(0.5*(t[i+1]+t[i]) - 0.5*(t[i]-t[i-1])))
    t_grid.append(t[i])
  t_grid = jnp.array(t_grid)
  acc_grid = jnp.stack(acc_grid)
  
  def potential(t, x):
    ids = jnp.argmin(jnp.abs(t - t_grid[None,:]), axis=1)
    out = -(x*acc_grid[ids]).sum(1, keepdims=True)
    return out
    
  return potential

# def get_physical_potential(config):
#   init_key = random.PRNGKey(0)
#   X, _ = datasets.get_data(config, init_key)
#   t = np.linspace(0.0, 1.0, len(X)).tolist()
#   if config.data.test_id is not None:
#     assert config.data.test_id < (len(X)-1) and config.data.test_id > 0
#     X_test = X.pop(config.data.test_id)
#     t_test = t.pop(config.data.test_id)
#     X_train = X[config.data.test_id-1]
#     t_train = t[config.data.test_id-1]
#   else:
#     X_test = X[1]
#     t_test = t[1]
#     X_train = X[0]
#     t_train = t[0]
#   if type(X_train) is list:
#     X_train = np.concatenate(X_train, axis=0)
#   X_train = jnp.array(X_train)
  
#   def potential(t, x):
#     dists = jnp.linalg.norm(x[:,None,:] - X_train[None,:,:], axis=-1)
#     ids = jnp.argsort(dists)[:,:10]
#     neigbours = jnp.array(X_train[ids])
#     neigbours_dists = ((neigbours - x[:,None,:])**2).sum(-1)
#     out = neigbours_dists.sum(1, keepdims=True)
#     # return jnp.minimum(out, 100)
#     return -out
    
#   return potential