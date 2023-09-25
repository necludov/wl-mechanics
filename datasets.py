import jax

from jax import numpy as jnp
import numpy as np

import scanpy as sc
import ot

def get_batch_iterator(config, init_key, eval=False):
  batch_size = config.eval.batch_size if eval else config.train.batch_size
  X, inv_scaler = get_data(config, init_key)
  t = np.linspace(0.0, 1.0, len(X)).tolist()
  if config.data.test_id is not None:
    assert config.data.test_id < (len(X)-1) and config.data.test_id > 0
    X_test = X.pop(config.data.test_id)
    t_test = t.pop(config.data.test_id)
    X_train = X[config.data.test_id-1]
    t_train = t[config.data.test_id-1]
  else:
    X_test = X[1]
    t_test = t[1]
    X_train = X[0]
    t_train = t[0]
    
  def eval_iterator(key):
    return (jnp.array(X_train), t_train, jnp.array(X_test), t_test)
    
  if eval:
    return eval_iterator, inv_scaler
    
  @jax.jit
  def linear_train_iterator(key):
    keys = jax.random.split(key, len(X))
    x_batch = jnp.zeros((batch_size, len(X), config.data.dim))
    t_batch = jnp.zeros((batch_size, len(X), 1))
    for i in range(len(X)):
      x_batch = x_batch.at[:,i,:].set(jax.random.choice(keys[i], X[i], (batch_size,), replace=False))
      t_batch = t_batch.at[:,i,:].set(t[i])
    x_batch = x_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(X),
                              config.data.dim)
    t_batch = t_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(t),
                              1)
    return (t_batch, x_batch)
  
  if config.interpolant == 'linear':
    return linear_train_iterator, inv_scaler
  
  log_plans = []
  for i in range(len(X)-1):
    a, b = ot.unif(X[i].shape[0]), ot.unif(X[i+1].shape[0])
    M = ot.dist(X[i], X[i+1], metric='euclidean')
    plan = ot.emd(a,b,M,numItermax=1e7)
    log_plans.append(jnp.array(np.log(plan/plan.sum(1, keepdims=True))))
    
  for i in range(len(X)):
    X[i] = jnp.array(X[i])
  
  @jax.jit
  def ot_train_iterator(key):
    keys = jax.random.split(key, len(X))
    x_batch = jnp.zeros((batch_size, len(X), config.data.dim))
    t_batch = jnp.zeros((batch_size, len(X), 1))
    for i in range(len(X)):
      if i == 0:
        ids = jax.random.categorical(keys[i], np.zeros((X[0].shape[0],)), shape=(batch_size,))
      else:
        ids = jax.random.categorical(keys[i], log_plans[i-1][ids], axis=1, shape=(batch_size,))
      x_batch = x_batch.at[:,i,:].set(X[i][ids])
      t_batch = t_batch.at[:,i,:].set(t[i])
    
    x_batch = x_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(X),
                              config.data.dim)
    t_batch = t_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(t),
                              1)
    return (t_batch, x_batch)
  
  if config.interpolant == 'ot':
    return ot_train_iterator, inv_scaler
  
  raise NotImplementedError(f'{config.interpolant} is not implemented')

from sklearn.preprocessing import StandardScaler

def get_data(config, init_key):
  if config.data.name == 'embrio':
    adata = sc.read_h5ad("/h/kirill/wl-mechanics/assets/ebdata_v3.h5ad")
    adata.obs["day"] = adata.obs["sample_labels"].cat.codes
  elif config.data.name == 'cite':
    adata = sc.read_h5ad("/h/kirill/wl-mechanics/assets/op_cite_inputs_0.h5ad")
  elif config.data.name == 'multi':
    adata = sc.read_h5ad("/h/kirill/wl-mechanics/assets/op_train_multi_targets_0.h5ad")
  elif config.data.name == 'toy':
    return get_toy_data(config, init_key)
  else:
    NotImplementedError(f'config.data.name: {config.data.name} is not implemented')
  times = adata.obs["day"].unique()
  coords = adata.obsm["X_pca"][:,:config.data.dim]
  if config.data.whiten:
    mu = coords.mean(axis=0, keepdims=True)
    sigma = coords.std(axis=0, keepdims=True)
    coords = (coords - mu) / sigma
    inv_scaler = lambda _x: _x
  else:
    mu = coords.mean(axis=0, keepdims=True)
    sigma = np.max(coords.std(axis=0, keepdims=True))
    coords = (coords - mu) / sigma
    inv_scaler = lambda _x: _x*sigma + mu
  adata.obsm["X_pca_standardized"] = coords
  X = [
    adata.obsm["X_pca_standardized"][adata.obs["day"] == t]
    for t in times
  ]
  return X, inv_scaler


def get_toy_data(config, init_key):
  init_key = jax.random.split(init_key)
  DS = 1_000
  sigma = 1e-1
  X_init = (jnp.ones((DS//8, 8))*(2*jnp.pi*jnp.arange(8)/8).reshape(1,-1)).reshape(-1,1)
  X_init = jnp.concatenate([jnp.cos(X_init), jnp.sin(X_init)], 1)
  X_final = 2*X_init
  X_init += sigma*jax.random.normal(init_key[0], shape=(DS, 2))
  X_final += sigma*jax.random.normal(init_key[1], shape=(DS, 2))
  return [X_init, X_final]
