## Utility functions for PyTorch CA implementation

import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import PIL
import tqdm

import matplotlib.pyplot as plt
from IPython.display import Image


TARGET_SIZE = 40

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
  return torch.clamp(x[..., 3:4], 0.0, 1.0)

def to_rgb(x):
  # assume rgb premultiplied by alpha
  rgb, a = x[..., :3], to_alpha(x)
  return 1.0-a+rgb

def visualize_batch(x0, x, step_i):
  vis0 = np.hstack(to_rgb(x0).numpy())
  vis1 = np.hstack(to_rgb(x).numpy())
  vis = np.vstack([vis0, vis1])
  # imwrite('train_log/batches_%04d.jpg'%step_i, vis)
  print('batch (before/after):')
  imshow(vis)

def plot_loss(loss_log):
  plt.figure(figsize=(10, 4))
  plt.title('Loss history (log10)')
  plt.plot(np.log10(loss_log), '.', alpha=0.1)
  plt.show()

## Additional utilities

def simulate_model(model, init, n_steps, device=torch.device('cuda')):
    """Runs the simulation for ca model `models` for n_steps, starting
        from initial condition `init`.
    """
    with torch.no_grad():
        x, model = init.to(device), model.to(device)
        for _ in tqdm.trange(n_steps):
            x = model(x)
        
        # If batch size is 1, use matplotlib's imshow
        if init.shape[0] == 1:
            
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(init.detach().cpu()[0, ..., :4])
            ax2.imshow(x.detach().cpu()[0, ..., :4])
            fig.show()            
        else:
            visualize_batch(init.detach().cpu(), x.detach().cpu(), n_steps)
        # return x.detach()

def save_ca_model(model, model_name):
    """Save model state dict as model_name in models folder
    
    :param model: trained PyTorch model object
    :param model_name: str, name under which to save the model
    """
    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(model.state_dict(), "./models/{}.pth".format(model_name))
    print("Saved model {} to disk".format(model_name))
    
def load_ca_model(model, model_name, device=None, *args, **kwargs):
    """Load model_name from models folder in working directory
    
    :param model: PyTorch model class name (not instantiatied object)
    :param model_name: str, name of saved model
    :return: PyTorch model object loaded from disk
    """
    ca = model(*args, **kwargs)
    if device:
        ca.load_state_dict(torch.load("./models/{}.pth".format(model_name), map_location=device))
    else:
        ca.load_state_dict(torch.load("./models/{}.pth".format(model_name)))
    ca.eval()
    return ca

def show_weights(ca):
    """Show weights for original model implementation.
        Note: assumes only two weight matrices and two bias vectors.
        
        :param ca: ca model for which to show weights
    """
    
    weights = []
    biases = []
    with torch.no_grad():
        for i, m in enumerate(ca.modules()):
            if isinstance(m, nn.Conv2d):
                weight = m.state_dict()['weight']
                bias = m.state_dict()['bias']
                weights.append(weight)
                biases.append(bias)

    # Get individual weights
    weights1 = weights[0].detach().cpu()
    weights2 = weights[1].detach().cpu()
    bias1 = biases[0].detach().cpu()
    bias2 = biases[1].detach().cpu()

    # Do plots
    fig, axs = plt.subplots(1, 4, figsize=(22, 4))
    axs[0].hist(weights1.flatten(), alpha=0.7, bins=100)
    axs[1].hist(bias1.flatten(), alpha=0.7, bins=100)
    axs[2].hist(weights2.flatten(), alpha=0.7, bins=100)
    axs[3].hist(bias2.flatten(), alpha=0.7, bins=100)
    fig.show()
    print(weights1.shape, bias1.shape, weights2.shape, bias2.shape)
    # return weights1.shape, bias1.shape, weights2.shape, bias2.shape
    
    
#     from collections import namedtuple
#     def make_asymmetric_seed(r, target_img=torch.tensor(target_img), p=TARGET_PADDING):
#         """Make 3 non-collinear points, distributed uniformly on a 
#             circular edge of predefined radius"""
#         pad_target = torch.nn.functional.pad(target_img, (0, 0, p, p, p, p))
#         h, w = pad_target.shape[:2]
#         Point = namedtuple('Point', ['x', 'y'])
#         center = Point(h//2, w//2)
#         print(center)
#         C = Point(center.x - r, center.y)
#         A = Point(center.x - r*math.cos(math.pi/6), center.y + r*math.sin(math.pi/6))
#         B = Point(center.x + r*math.cos(math.pi/6), A.y)
#         print(C, A, B)

#         angles = [0, 2/3*math.pi, 4/3*math.pi]
#         cx, cy = center.x, center.y
#         # Calculate the coordinates of each point
#         points = [(cx + r*math.cos(angle), cy + r*math.sin(angle)) for angle in angles]

#         return center, points

#         # C = (center[0]
#         # seed = torch.zeros(h, w, CHANNEL_N, dtype=torch.float32)

#     center, (a,b,c) = make_asymmetric_seed(6)
#     a, b, c