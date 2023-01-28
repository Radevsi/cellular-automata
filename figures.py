
import glob
import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import numpy as np
import tqdm

import model
import utils

# From original Colab
class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

class FigGen:
  def __init__(self, ca, path):
    self.models = []
    self.ca = ca
    self.path = path

  def training_progress_checkpoints(self, damage_n, channel_n, steps):

    print("IN checkpoints FUNCTION")
    for i in steps:
      self.ca.load_weights(f'{self.path}/train_log/%04d'%i)
      self.models.append(self.ca)

    out_fn = f'{self.path}/train_steps_damage_%d.mp4'%damage_n
    x = np.zeros([len(self.models), 72, 72, channel_n], np.float32)
    x[..., 36, 36, 3:] = 1.0
    with VideoWriter(out_fn) as vid:
      for i in tqdm.trange(500):
        vis = np.hstack(model.to_rgb(x))
        vid.add(utils.zoom(vis, 2))
        for ca, xk in zip(self.models, x):
          xk[:] = ca(xk[None,...])[0]
    
    # Make a VideoFileClip object and then write it 
    clip = mvp.VideoFileClip(out_fn)
    clip.write_videofile(f'{out_fn}')
    
  def training_progress_batches(self):
    frames = sorted(glob.glob(f'{self.path}/train_log/batches_*.jpg'))
    mvp.ImageSequenceClip(frames, fps=10.0).write_videofile(f'{self.path}/batches.mp4')

  def pool_contents(self):
    frames = sorted(glob.glob(f'{self.path}/train_log/*_pool.jpg'))[:80]
    mvp.ImageSequenceClip(frames, fps=20.0).write_videofile(f'{self.path}/pool.mp4')
