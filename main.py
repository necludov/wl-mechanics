from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os

import run_lib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("mode", "train", "train, eval, 5seeds")
flags.mark_flags_as_required(["workdir", "config"])

def launch(argv):
  tf.config.experimental.set_visible_devices([], "GPU")
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  if FLAGS.mode == 'train':
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == 'eval':
    run_lib.eval(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == '5seeds':
    config = FLAGS.config
    for seed in range(config.seed,config.seed+5):
      config.seed = seed
      for test_id in [1,2]:
        config.data.test_id = test_id
        run_lib.train(config, FLAGS.workdir)
  else:
    NotImplementedError(f'FLAGS.mode: {FLAGS.mode} is not implemented')

if __name__ == "__main__":
  app.run(launch)
