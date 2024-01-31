import gc
import os
import functools
import json
import io

import ot
import wandb
from tqdm.auto import tqdm, trange

import jax
import flax
import numpy as np
import flax.jax_utils as flax_utils
import tensorflow as tf
from jax import random, jit
from jax import numpy as jnp
from flax.training import checkpoints

import losses
import datasets
import train_utils as tutils
import eval_utils as eutils

from models import utils as mutils
from models import mlp


def train(config, workdir):
  print(f'running config.seed: {config.seed}', flush=True)
  key = random.PRNGKey(config.seed)
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tf.io.gfile.makedirs(checkpoint_dir)

  key, *init_key = random.split(key, 3)
  # init model s
  model_s, _, initial_params = mutils.init_model_s(init_key[0], config.model_s)
  optimizer_s = tutils.get_optimizer(config.optimizer_s)
  opt_state_s = optimizer_s.init(initial_params)
  time_sampler, init_sampler_state = tutils.get_time_sampler(config)

  state_s = mutils.State(step=1, opt_state=opt_state_s,
                         model_params=initial_params,
                         ema_rate=config.model_s.ema_rate,
                         params_ema=initial_params,
                         sampler_state=init_sampler_state,
                         key=key, wandbid=np.random.randint(int(1e7),int(1e8)))
  state_s = checkpoints.restore_checkpoint(checkpoint_dir, state_s, prefix='chkpt_s_')
  initial_step = int(state_s.step)
  key = state_s.key

  # init model q
  model_q, _, initial_params = mutils.init_model_q(init_key[1], config.model_q)
  optimizer_q = tutils.get_optimizer(config.optimizer_q)
  opt_state_q = optimizer_q.init(initial_params)

  state_q = mutils.State(step=state_s.step, opt_state=opt_state_q,
                         model_params=initial_params,
                         ema_rate=config.model_q.ema_rate,
                         params_ema=initial_params,
                         sampler_state=init_sampler_state,
                         key=state_s.key, wandbid=state_s.wandbid)
  state_q = checkpoints.restore_checkpoint(checkpoint_dir, state_q, prefix='chkpt_q_')

  if jax.process_index() == 0:
    wandb.init(id=str(state_s.wandbid), 
               project='single-cell-' + config.data.name, 
               resume="allow",
               config=json.loads(config.to_json_best_effort()))
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = str(state_s.wandbid)
  
  # init train step
  loss_fn = losses.get_loss(config, model_s, model_q, time_sampler, train=True)
  step_fn = tutils.get_step_fn(config, optimizer_s, optimizer_q, loss_fn)
  step_fn = jax.pmap(functools.partial(jax.lax.scan, step_fn), axis_name='batch')
  
  # init dataloaders
  key, *init_key = random.split(key, 3)
  batch_iterator, inv_scaler = datasets.get_batch_iterator(config, init_key[0])
  val_iterator, _ = datasets.get_batch_iterator(config, init_key[1], eval=True, val=True)
  test_iterator, _ = datasets.get_batch_iterator(config, init_key[1], eval=True)
  
  # init eval generators
  pairwise_dist = jax.jit(lambda _x,_y: jnp.linalg.norm(_x[:,None,:]-_y[None,:,:], axis=-1))
  ode_generator, ot_generator = eutils.get_generator(model_s, config)
  ode_generator, ot_generator = jax.jit(ode_generator), jax.jit(ot_generator)

  # run train
  # assert (config.train.n_iters % config.train.save_every) == 0

  state_s = flax_utils.replicate(state_s)
  state_q = flax_utils.replicate(state_q)
  key = jax.random.fold_in(key, jax.process_index())
  for step in range(initial_step, config.train.n_iters+1, config.train.n_jitted_steps):
    key, batch_key = random.split(key)
    batch = batch_iterator(batch_key)
    key, *next_key = random.split(key, num=jax.local_device_count() + 1)
    next_key = jnp.asarray(next_key)
    (_, state_s, state_q), (total_loss, metrics) = step_fn((next_key, state_s, state_q), batch)
    total_loss = flax.jax_utils.unreplicate(total_loss).mean()
    
    if (step % config.train.log_every == 0) and (jax.process_index() == 0):
      logging_dict = dict(total_loss=total_loss.mean().item())
      for k in metrics:
        logging_dict[k] = metrics[k].mean().item()
        wandb.log(logging_dict, step=step)

    if (step % config.train.save_every == 0) and (jax.process_index() == 0):
      saved_state = flax_utils.unreplicate(state_s)
      saved_state = saved_state.replace(key=key)
      checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                  step=step // config.train.save_every,
                                  keep=50, prefix='chkpt_s_')
      saved_state = flax_utils.unreplicate(state_q)
      saved_state = saved_state.replace(key=key)
      checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                  step=step // config.train.save_every,
                                  keep=50, prefix='chkpt_q_')

    if (step % config.train.eval_every == 0) and (jax.process_index() == 0):
      X_init, t_init, X_end, t_end = val_iterator()
      for i in range(len(X_init)):
        key, *eval_keys = random.split(key, 4)
        (ode_solution, weights), ode_steps = ode_generator(eval_keys[1], flax_utils.unreplicate(state_s), (X_init[i], t_init[i], X_end[i], t_end[i]))
        # (ot_solution, _), ot_steps = ot_generator(eval_keys[2], flax_utils.unreplicate(state_s), (X_train, t_train, X_test, t_test))
        if config.metric == 'w1':
          metric = eutils.get_w1(pairwise_dist(inv_scaler(ode_solution), inv_scaler(X_end[i])), weights)
        if config.metric == 'w2':
          metric = np.sqrt(eutils.get_w1(pairwise_dist(inv_scaler(ode_solution), inv_scaler(X_end[i]))**2, weights))
        wandb.log({'metric_val_%d' % (i + 1): metric}, step=step)
        
      X_init, t_init, X_end, t_end = test_iterator()
      for i in range(len(X_init)):
        key, *eval_keys = random.split(key, 4)
        (ode_solution, weights), ode_steps = ode_generator(eval_keys[1], flax_utils.unreplicate(state_s), (X_init[i], t_init[i], X_end[i], t_end[i]))
        # (ot_solution, _), ot_steps = ot_generator(eval_keys[2], flax_utils.unreplicate(state_s), (X_train, t_train, X_test, t_test))
        if config.metric == 'w1':
          metric = eutils.get_w1(pairwise_dist(inv_scaler(ode_solution), inv_scaler(X_end[i])), weights)
        if config.metric == 'w2':
          metric = np.sqrt(eutils.get_w1(pairwise_dist(inv_scaler(ode_solution), inv_scaler(X_end[i]))**2, weights))
        wandb.log({'metric_test_%d' % (i + 1): metric} , step=step)
        
        if weights is not None:
          weights /= weights.sum()
          ids = np.random.choice(len(ode_solution), len(X_end[i]), p=np.array(weights))
          ode_solution = ode_solution[ids]
        mmd = eutils.compute_scalar_mmd(inv_scaler(ode_solution), inv_scaler(X_end[i]))
        wandb.log({'mmd_test_%d' % (i + 1): mmd} , step=step)
  wandb.finish()


def eval(config, workdir):
  pairwise_dist = jax.jit(lambda _x,_y: jnp.linalg.norm(_x[:,None,:]-_y[None,:,:], axis=-1))
  
  def midway_preds(x, y):
    a, b = ot.unif(x.shape[0]), ot.unif(y.shape[0])
    M = pairwise_dist(jnp.array(x), jnp.array(y))
    M = np.array(M).astype(np.float64)
    plan = ot.emd(a, b, M, numItermax=1e7)
    ids = np.argmax(plan, axis=1)
    return 0.5*(x + y[ids])
  
  key = random.PRNGKey(0)
  X, inv_scaler = datasets.get_data(config, key)
  assert len(X) > 2
  test_ids = list(range(1,len(X)-1))
  w1 = np.zeros(len(test_ids))
  print(len(X), len(w1))
  for i in range(len(test_ids)):
    test_marginal = test_ids[i]
    preds = midway_preds(X[test_marginal-1], X[test_marginal+1])
    w1[i] = eutils.get_w1(pairwise_dist(jnp.array(preds), jnp.array(X[test_marginal])))
  
  print(w1, w1.mean())
