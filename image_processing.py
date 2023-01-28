## image_processing.py
# Processes any input image that is not an emoji 
# and returns the appropriate target_img
# to be used.

import PIL.Image
import numpy as np
import math
import tensorflow_addons as tfa

# Load an image locally, by initializing a white background with alpha values
# of zero to denote "dead" cells.
def load_local_image(image_name, max_size, threshold=0.1, sigma=1.0):
  path = f'images/{image_name}.png' 
  img = PIL.Image.open(path)
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  orig_img = img.copy()

  # Concatenate alpha channel initialized with zeros
  target_img = np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)

  for i in range(target_img.shape[0]):
    for j in range(target_img.shape[1]):
      if (target_img[i][j][0] > 0.95 and target_img[i][j][1] > 0.95 and target_img[i][j][2] > 0.95):
        target_img[i][j][0] = 0
        target_img[i][j][1] = 0
        target_img[i][j][2] = 0

        # Append the alpha channel
      else:
        l2_norm = math.sqrt(target_img[i][j][0]**2 + target_img[i][j][1]**2 + target_img[i][j][2]**2)
        if l2_norm > threshold:
          target_img[i][j][3] = 1

  # Perform guassian blurring to smooth out boundaries
  alpha_channel = tfa.image.gaussian_filter2d(image=target_img[..., 3], sigma=sigma, padding='CONSTANT')
  target_img[..., 3] = alpha_channel
  
  # Premultiply RGB by Alpha
  target_img[..., :3] *= target_img[..., 3:]

  return target_img, alpha_channel, orig_img

# load_path = 'mozart1.png'
# target_img, alpha_channel, orig_img = load_local_image(load_path, 48)

# Helper code to load "alive" image data (no white background).
def load_alive_image(image_name, max_size):
  path = f'images/{image_name}.png'
  img = PIL.Image.open(path)
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  orig_img = img.copy()

  # Concatenate alpha channel initialized with zeros
  target_img = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)

  # Premultiply RGB by Alpha
  target_img[..., :3] *= target_img[..., 3:]

  return target_img, target_img[..., 3], orig_img