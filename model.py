# Defines the CA Model and provides utilities

print('\n...........................IN model.py...........................')

import sys
import io
import PIL.Image, PIL.ImageDraw
import requests
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Conv2D

import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'

def load_image(url, max_size=40):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img

def load_emoji(emoji):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url)

def to_rgba(x):
  return x[..., :4]

def to_alpha(x):
  return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
  rgb, a = x[..., :3], to_alpha(x)
  return 1.0-a+rgb

def get_living_mask(x):
  alpha = x[:, :, :, 3:4]
  return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

def make_seed(size, channel_n, n=1):
  x = np.zeros([n, size, size, channel_n], np.float32)
  x[:, size//2, size//2, 3:] = 1.0
  return x


class CAModel(tf.keras.Model):

  def __init__(self, channel_n, hidden_size, fire_rate, step_size=1.0):
    super().__init__()
    self.channel_n = channel_n
    self.fire_rate = fire_rate
    self.step_size = step_size

    self.dmodel = tf.keras.Sequential()
    if type(hidden_size) == int:
      self.dmodel.add(Conv2D(hidden_size, 1, activation=tf.nn.relu))
    elif type(hidden_size) == list:
      for hs in hidden_size:
        self.dmodel.add(Conv2D(hs, 1, activation=tf.nn.relu))
    else:
      print("WARNING: Tried to pass in a hidden size that is neither an int nor a list")
      sys.exit(1)

    self.dmodel.add(Conv2D(self.channel_n, 1, activation=None, kernel_initializer=tf.zeros_initializer))

    # self.dmodel = tf.keras.Sequential([
    #       Conv2D(hidden_size, 1, activation=tf.nn.relu),
    #       Conv2D(self.channel_n, 1, activation=None,
    #           kernel_initializer=tf.zeros_initializer),
    # ])

    self(tf.zeros([1, 3, 3, channel_n]))  # dummy call to build the model

  @tf.function
  def perceive(self, x, angle=0.0):
    identify = np.float32([0, 1, 0])
    identify = np.outer(identify, identify)
    dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
    dy = dx.T
    c, s = tf.cos(angle), tf.sin(angle)
    kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
    kernel = tf.repeat(kernel, self.channel_n, 2)
    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    return y

  @tf.function
  def call(self, x, fire_rate=None, angle=0.0, step_size=None):
    pre_life_mask = get_living_mask(x)
    
    if step_size is None:
      step_size = self.step_size
    y = self.perceive(x, angle)
    dx = self.dmodel(y)*step_size
    if fire_rate is None:
      fire_rate = self.fire_rate
    update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
    x += dx * tf.cast(update_mask, tf.float32)

    post_life_mask = get_living_mask(x)
    life_mask = pre_life_mask & post_life_mask
    return x * tf.cast(life_mask, tf.float32)
