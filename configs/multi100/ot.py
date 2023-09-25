import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.seed = 0
  config.loss = 'am'
  config.interpolant = 'linear'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.task = 'OT'
  data.name = 'multi'
  data.dim = 100
  data.whiten = False
  data.test_id = 1
  data.t_0, data.t_1 = 0.0, 1.0

  # models
  config.model_s = model_s = ml_collections.ConfigDict()
  model_s.input_dim = data.dim
  model_s.name = 'mlp_s'
  model_s.ema_rate = 0.999
  model_s.nonlinearity = 'swish'
  model_s.nf = 512
  model_s.n_layers = 3
  model_s.skip = False
  model_s.embed_time = True
  model_s.dropout = 0.0

  config.model_q = model_q = ml_collections.ConfigDict()
  model_q.input_dim = data.dim
  model_q.n_marginals = 4 if data.test_id is None else 3
  model_q.name = 'mlp_q'
  model_q.ema_rate = 0.999
  model_q.nonlinearity = 'swish'
  model_q.nf = 512
  model_q.n_layers = 1
  model_q.skip = False
  model_q.indicator = False
  model_q.dropout = 0.0

  # opts
  config.optimizer_s = optimizer_s = ml_collections.ConfigDict()
  optimizer_s.name = 'adamw'
  optimizer_s.lr = 2e-4
  optimizer_s.beta1 = 0.9
  optimizer_s.eps = 1e-8
  optimizer_s.warmup = 5_000
  optimizer_s.grad_clip = 1.

  config.optimizer_q = optimizer_q = ml_collections.ConfigDict()
  optimizer_q.name = 'adamw'
  optimizer_q.lr = 2e-4
  optimizer_q.beta1 = 0.9
  optimizer_q.eps = 1e-8
  optimizer_q.warmup = 5_000
  optimizer_q.grad_clip = 1.
  
  # training
  config.train = train = ml_collections.ConfigDict()
  train.batch_size = 512
  train.n_gradient_steps = 10
  train.step_size = 1e-2
  train.n_jitted_steps = 1
  train.n_iters = 100_000
  train.save_every = 200_000
  train.eval_every = 10_000
  train.log_every = 50

  # evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.batch_size = 128
  eval.num_samples = 500
  eval.use_ema = True

  return config
