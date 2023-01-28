# Module for training the CA model

print('\n...........................IN train.py...........................')

import os
import json
import numpy as np
import matplotlib.pylab as pl

import tensorflow as tf

from utils import imwrite, tile2d 
os.environ['FFMPEG_BINARY'] = 'ffmpeg'

# Initialize training
from model import to_rgb, to_rgba
from utils import export_ca_to_webgl_demo

from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

# Train Utilities (SamplePool, Model Export, Damage)
class SamplePool:
  def __init__(self, *, _parent=None, _parent_idx=None, **slots):
    self._parent = _parent
    self._parent_idx = _parent_idx
    self._slot_names = slots.keys()
    self._size = None
    for k, v in slots.items():
      if self._size is None:
        self._size = len(v)
      assert self._size == len(v)
      setattr(self, k, np.asarray(v))

  def sample(self, n):
    idx = np.random.choice(self._size, n, False)
    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

@tf.function
def make_circle_masks(n, h, w):
  x = tf.linspace(-1.0, 1.0, w)[None, None, :]
  y = tf.linspace(-1.0, 1.0, h)[None, :, None]
  center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
  r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
  x, y = (x-center[0])/r, (y-center[1])/r
  mask = tf.cast(x*x+y*y < 1.0, tf.float32)
  return mask

def export_model(ca, base_fn, channel_n):
  ca.save_weights(base_fn)

  cf = ca.call.get_concrete_function(
      x=tf.TensorSpec([None, None, None, channel_n]),
      fire_rate=None,
      # fire_rate=tf.constant(0.5),
      angle=tf.constant(0.0),
      step_size=None) # Use step_size passed into model initialization
      # step_size=tf.constant(1.0))
  cf = convert_to_constants.convert_variables_to_constants_v2(cf)
  graph_def = cf.graph.as_graph_def()
  graph_json = MessageToDict(graph_def)
  graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
  model_json = {
      'format': 'graph-model',
      'modelTopology': graph_json,
      'weightsManifest': [],
  }
  with open(base_fn+'.json', 'w') as f:
    json.dump(model_json, f)

def generate_pool_figures(pool, step_i, path=''):
  tiled_pool = tile2d(to_rgb(pool.x[:49]))
  fade = np.linspace(1.0, 0.0, 72)
  ones = np.ones(72) 
  tiled_pool[:, :72] += (-tiled_pool[:, :72] + ones[None, :, None]) * fade[None, :, None] 
  tiled_pool[:, -72:] += (-tiled_pool[:, -72:] + ones[None, :, None]) * fade[None, ::-1, None]
  tiled_pool[:72, :] += (-tiled_pool[:72, :] + ones[:, None, None]) * fade[:, None, None]
  tiled_pool[-72:, :] += (-tiled_pool[-72:, :] + ones[:, None, None]) * fade[::-1, None, None]
  imwrite(f'{path}/train_log/%04d_pool.jpg'%step_i, tiled_pool)

def visualize_batch(x0, x, step_i, path=''):
  vis0 = np.hstack(to_rgb(x0).numpy())
  vis1 = np.hstack(to_rgb(x).numpy())
  vis = np.vstack([vis0, vis1])
  imwrite(f'{path}/train_log/batches_%04d.jpg'%step_i, vis)
  # print('batch (before/after):')
  # imshow(vis)

def plot_loss(loss_log, channel_n, hidden_size, image_name, target_size, 
              save=False, path=''):   
  pl.figure(figsize=(10, 4))
  pl.title(f'Log10 Loss For {image_name}.png With Max Size {target_size}')
  pl.ylabel('Loss history (log10)')
  pl.xlabel('Time')
  pl.plot(np.log10(loss_log), '.', alpha=0.1, label=f'{channel_n} Channels, {hidden_size} Hidden Size')
  pl.ylim(top=0) # ignore outliers
  pl.legend()
  # pl.show()
  if save:
    pl.savefig(f'{path}/loss_plot-{channel_n}-{hidden_size}.png')
  else:
    pl.draw()
  return pl
  
def loss_f(x, pad_target):
  return tf.reduce_mean(tf.square(to_rgba(x)-pad_target), [-2, -3, -1])

@tf.function
def train_step(ca, x, trainer, pad_target):
  """Applies single training step. Unchanged."""
  iter_n = tf.random.uniform([], 64, 96, tf.int32)
  with tf.GradientTape() as g:
    for i in tf.range(iter_n):
      x = ca(x)
    loss = tf.reduce_mean(loss_f(x, pad_target))
  grads = g.gradient(loss, ca.weights)
  grads = [g/(tf.norm(g)+1e-8) for g in grads]
  # trainer.apply_gradients(zip(grads, ca.weights))
  return x, loss, grads


def train_ca(ca, image_name, target_size, target_img, channel_n, hidden_size, 
              target_padding, batch_size, pool_size, use_pattern_pool, damage_n, 
              trainer=None, steps=8000, lr=2e-3, path='', make_pool=False):
  """
  Main training function. 
  Equivalent to 'Training Loop' in Colab Notebook.
  """

  p = target_padding

  if trainer is None:
    lr_sched=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [2000], [lr, lr*0.1])
    trainer=tf.keras.optimizers.Adam(lr_sched)

  loss_log = []
  pad_target = tf.pad(target_img, [(p, p), (p, p), (0, 0)])
  h, w = pad_target.shape[:2]
  seed = np.zeros([h, w, channel_n], np.float32)
  seed[h//2, w//2, 3:] = 1.0

  # loss0 = loss_f(seed).numpy()
  pool = SamplePool(x=np.repeat(seed[None, ...], pool_size, 0))

  for i in range(steps+1):
    if use_pattern_pool:
      batch = pool.sample(batch_size)
      x0 = batch.x
      loss_rank = loss_f(x0, pad_target).numpy().argsort()[::-1]
      x0 = x0[loss_rank]
      x0[:1] = seed
      if damage_n:
        damage = 1.0-make_circle_masks(damage_n, h, w).numpy()[..., None]
        x0[-damage_n:] *= damage
    else:
      x0 = np.repeat(seed[None, ...], batch_size, 0)

    x, loss, grads = train_step(ca, x0, trainer, pad_target)
    trainer.apply_gradients(zip(grads, ca.weights)) # gradient update

    if use_pattern_pool:
      batch.x[:] = x
      # batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.numpy())
    
    if make_pool and step_i%10 == 0:
      generate_pool_figures(pool, step_i, path=path)
    if step_i%100 == 0:
      visualize_batch(x0, x, step_i, path=path)
      plot_loss(loss_log, channel_n, hidden_size, image_name, target_size, path=path)
      export_model(ca, f'{path}/train_log/%04d'%step_i, channel_n)

    print('\r step: %d/%d, log10(loss): %.3f'%(len(loss_log), steps, np.log10(loss)), end='')

  # Save the loss_log array
  with open(path+'/loss_log.npy', 'wb') as file:
    np.save(file, loss_log)
    
  # Locally save final loss plot
  # plot_loss(loss_log, save=True, path=path)
  plot_loss(loss_log, channel_n, hidden_size, image_name, target_size, path=path)

  with open(f'{path}/webgl_model.json', 'w') as outfile:
    outfile.write(export_ca_to_webgl_demo(ca))

  # Return the loss array
  return loss_log

  
