import jax

from jax import numpy as jnp
import numpy as np

import scanpy as sc
import ot


def get_batch_iterator(config, init_key, eval=False, val=False):
  batch_size = config.eval.batch_size if eval else config.train.batch_size
  X_train, X_test, X_val, inv_scaler, times = get_data(config, init_key)
  assert len(X_train) == len(X_test)
  print(times, 'times', flush=True)
  t = ((times - np.min(times))/(np.max(times)-np.min(times))).tolist()
  print(t, 'times normalized', flush=True)
    
  if config.data.name == '4i':
    def test_iterator():
      return ([X_test[0]], [t[0]], [X_test[-1]], [t[-1]])
    def val_iterator():
      return ([X_val[0]], [t[0]], [X_val[-1]], [t[-1]])
  elif config.data.test_id is not None:
    assert config.data.test_id < (len(X)-1) and config.data.test_id > 0
    def test_iterator():
      return ([X_train[config.data.test_id-1]], [t[config.data.test_id-1]], 
              [X_train.pop(config.data.test_id)], [t.pop(config.data.test_id)])
    def val_iterator():
      return ([X_train[config.data.test_id-1]], [t[config.data.test_id-1]], 
              [X_train.pop(config.data.test_id)], [t.pop(config.data.test_id)])
  else: 
    def test_iterator():
      return (X_test[:-1], t[:-1], X_test[1:], t[1:])
    def val_iterator():
      return (X_val[:-1], t[:-1], X_val[1:], t[1:])
  if eval:
    if val:
      return val_iterator, inv_scaler
    else:
      return test_iterator, inv_scaler
    
  @jax.jit
  def linear_train_iterator(key):
    keys = jax.random.split(key, len(X_train))
    x_batch = jnp.zeros((batch_size, len(X_train), config.data.dim))
    t_batch = jnp.zeros((batch_size, len(X_train), 1))
    for i in range(len(X_train)):
      x_batch = x_batch.at[:,i,:].set(jax.random.choice(keys[i], X_train[i], (batch_size,), replace=True))
      t_batch = t_batch.at[:,i,:].set(t[i])
    x_batch = x_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(X_train),
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
  for i in range(len(X_train)-1):
    a, b = ot.unif(X_train[i].shape[0]), ot.unif(X_train[i+1].shape[0])
    M = ot.dist(X_train[i], X_train[i+1], metric='euclidean')
    plan = ot.emd(a,b,M,numItermax=1e7)
    log_plans.append(jnp.array(np.log(plan/plan.sum(1, keepdims=True))))
    
  for i in range(len(X_train)):
    X_train[i] = jnp.array(X_train[i])
  
  @jax.jit
  def ot_train_iterator(key):
    keys = jax.random.split(key, len(X_train))
    x_batch = jnp.zeros((batch_size, len(X_train), config.data.dim))
    t_batch = jnp.zeros((batch_size, len(X_train), 1))
    for i in range(len(X_train)):
      if i == 0:
        ids = jax.random.categorical(keys[i], np.zeros((X_train[0].shape[0],)), shape=(batch_size,))
      else:
        ids = jax.random.categorical(keys[i], log_plans[i-1][ids], axis=1, shape=(batch_size,))
      x_batch = x_batch.at[:,i,:].set(X[i][ids])
      t_batch = t_batch.at[:,i,:].set(t[i])
    
    x_batch = x_batch.reshape(jax.local_device_count(),
                              config.train.n_jitted_steps, 
                              batch_size//jax.local_device_count(),
                              len(X_train),
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

def get_data(config, init_key):
  if config.data.name == '4i':
    return get_rna_data(config, init_key)
  if config.data.name == 'rna':
    return get_rna_data(config, init_key)
  if config.data.name == 'toy':
    return get_toy_data(config, init_key)
  if config.data.name == 'embrio':
    return get_h5ad_data(config, init_key)
  if config.data.name == 'cite':
    return get_h5ad_data(config, init_key)
  if config.data.name == 'multi':
    return get_h5ad_data(config, init_key)
  NotImplementedError(f'config.data.name: {config.data.name} is not implemented')
  

def get_h5ad_data(config, init_key):
  if config.data.name == 'embrio':
    adata = sc.read_h5ad("assets/ebdata_v3.h5ad")
    adata.obs["day"] = adata.obs["sample_labels"].cat.codes
  elif config.data.name == 'cite':
    adata = sc.read_h5ad("assets/op_cite_inputs_0.h5ad")
  elif config.data.name == 'multi':
    adata = sc.read_h5ad("assets/op_train_multi_targets_0.h5ad")
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
  X_train, X_test, X_val = X, X, X
  times = np.linspace(0.0, 1.0, len(X)).tolist()
  return X_train, X_test, X_val, inv_scaler, times


def get_rna_data(config, init_key):
  def load_rna(filename):
    with np.load(filename) as data:
      t = data['ts']
      X = data['X'][:,:config.data.dim]
    return t, X
  t_train, X_train = load_rna(f'assets/train_{config.data.name}.npz')
  t_test, X_test = load_rna(f'assets/test_{config.data.name}.npz')
  t_val, X_val = load_rna(f'assets/val_{config.data.name}.npz')
  if config.data.whiten:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    X_val = (X_val - mu) / sigma
    inv_scaler = lambda _x: _x
  else:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = np.max(X_train.std(axis=0, keepdims=True))
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    X_val = (X_val - mu) / sigma
    inv_scaler = lambda _x: _x*sigma + mu
    
  times = np.unique(t_train)
  X_train = [jnp.array(X_train[t_train == t]) for t in times]
  X_test = [jnp.array(X_test[t_test == t]) for t in times]
  X_val = [jnp.array(X_val[t_val == t]) for t in times]
  return X_train, X_test, X_val, inv_scaler, times


def get_toy_data(config, init_key):
  if config.loss == 'am':
    return get_toy_data_for_ot(config, init_key)
  if config.loss == 'sb':
    return get_toy_data_for_ot(config, init_key)
  if config.loss == 'ubot':
    return get_toy_data_for_ubot(config, init_key)
  if config.loss == 'phot':
    return get_toy_data_for_phot(config, init_key)

def get_toy_data_for_ubot(config, init_key):
  init_key = jax.random.split(init_key)
  DS = 2_000
  sigma = 3e-1
  X_init = jnp.concatenate([-jnp.ones((DS,1)), jnp.zeros((DS,1))], 1)
  X_final = -X_init
  X_init += sigma*jax.random.normal(init_key[0], shape=(X_init.shape[0], 2))
  X_final += sigma*jax.random.normal(init_key[1], shape=(X_final.shape[0], 2))
  
  X_train = X_val = X_test = [X_init, X_final]
  inv_scaler = lambda _x: _x
  times = np.array([0.0, 1.0])
  return  X_train, X_test, X_val, inv_scaler, times

def get_toy_data_for_phot(config, init_key):
  init_key = jax.random.split(init_key)
  DS = 2_000
  sigma = 1e-1
  X_init = jnp.concatenate([-jnp.ones((DS,1)), jnp.zeros((DS,1))], 1)
  X_final = -X_init
  X_init += sigma*jax.random.normal(init_key[0], shape=(DS, 2))
  X_final += sigma*jax.random.normal(init_key[1], shape=(DS, 2))
  
  X_train = X_val = X_test = [X_init, X_final]
  inv_scaler = lambda _x: _x
  times = np.array([0.0, 1.0])
  return  X_train, X_test, X_val, inv_scaler, times

def get_toy_data_for_ot(config, init_key):
  init_key = jax.random.split(init_key)
  DS = 2_000
  sigma = 1e-1
  X_init = (jnp.ones((DS//8, 8))*(2*jnp.pi*jnp.arange(8)/8).reshape(1,-1)).reshape(-1,1)
  X_init = jnp.concatenate([jnp.cos(X_init), jnp.sin(X_init)], 1)
  X_final = 2*X_init
  X_init += sigma*jax.random.normal(init_key[0], shape=(DS, 2))
  X_final += sigma*jax.random.normal(init_key[1], shape=(DS, 2))
  X_train = X_val = X_test = [X_init, X_final]
  inv_scaler = lambda _x: _x
  times = np.array([0.0, 1.0])
  return  X_train, X_test, X_val, inv_scaler, times
