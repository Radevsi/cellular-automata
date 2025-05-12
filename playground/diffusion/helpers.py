import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import requests
import PIL
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython.display import Image



# Helpful utils 
TARGET_SIZE = 50

"Image processing functions "

def load_image(url, max_size=TARGET_SIZE):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img) / 255.0

  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:]
  return img

def load_emoji(emoji):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
  return load_image(url)

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))

def to_rgba(x):
  "This function used outside model, using original shaping conventions"
  return x[..., :4]

def get_living_mask(x):
  "This function used within model with PyTorch shaping conventions"
  alpha = x[:, 3:4, :, :]
  return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

def to_alpha(x):
  "Assume original TF shaping convention"
  return np.clip(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  # Assume rgb premultiplied by alpha
  rgb, a = x[..., :3], to_alpha(x)
  return 1.0-a+rgb

def visualize_batch(x0, x, step_i=0):
    if x0.shape[2] == x0.shape[3]: # if shape [B, C, H, W]
        x0 = x0.permute(0, 2, 3, 1).detach().cpu()
    if x.shape[2] == x.shape[3]: # if shape [B, C, H, W]
        x = x.permute(0, 2, 3, 1).detach().cpu()
    # print(f'shape of x: {x.shape}, shape of x0: {x0.shape}')
    vis0 = np.hstack(to_rgb(x0).numpy())
    vis1 = np.hstack(to_rgb(x).numpy())  
    vis = np.vstack([vis0, vis1])
    print('batch (before/after):')
    imshow(vis)

def plot_loss(loss_log):
  plt.figure(figsize=(10, 4))
  plt.title('Loss history (log10)')
  plt.plot(np.log10(loss_log), '.', alpha=0.1)
  plt.show()
   


def save_np_array(loss_log, filename, foldername="losses"):
    """Save loss_log array as filename in foldername folder
    
    :param loss_log: np.array, loss_log array
    :param filename: str, name under which to save loss_log array
    """
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    with open(f"./{foldername}/{filename}.npy", 'wb') as file:
        np.save(file, loss_log)
    print(f"Saved array under {foldername}/{filename} name to disk")
    
def load_np_array(filename, foldername="losses"):
    """Load `filename` from `foldername` folder in working directory
    
    :param filename: str, name of saved object
    :param foldername: str, name of folder from which to load `filename`
    :return: np.array loss_log array loaded from disk
    """
    with open(f"./{foldername}/{filename}.npy", 'rb') as file:
        return np.load(file, allow_pickle=True)




# Helpers from/for diffusion

def plot_images(images, cmap='viridis'):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
    ], dim=-2).permute(1, 2, 0).cpu(), cmap=cmap)
    plt.show()
    
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = PIL.Image.fromarray(ndarr)
    im.save(path)
    
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
# From texture blog
def imread(url, max_size=None, mode=None):
  if url.startswith(('http:', 'https:')):
    # wikimedia requires a user agent
    headers = {
      "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
    }
    r = requests.get(url, headers=headers)
    f = io.BytesIO(r.content)
  else:
    f = url
  img = PIL.Image.open(f)
  if max_size is not None:
    img.thumbnail((max_size, max_size), PIL.Image.Resampling.LANCZOS)
  if mode is not None:
    img = img.convert(mode)
  img = np.float32(img)/255.0
  return img