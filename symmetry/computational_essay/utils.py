## Utility functions for an Equivariant CA implementation in PyTorch

import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import PIL
import tqdm
import math

import matplotlib.pyplot as plt
from IPython.display import Image

import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

# This defines how pixelated the image will be. This is the original default
TARGET_SIZE = 40

"Image processing functions (mostly reimplemented in torch from original Distill blog)"

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

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

def to_rgba(x):
  "This function used outside model, using original shaping conventions"
  # if len(x.shape) == 3:
  #   return x[..., :4]
  # elif len(x.shape) == 4:
  #   return x[0, ..., :4]
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

def visualize_batch(x0, x, step_i=0, verbose=True):
  vis0 = np.hstack(to_rgb(x0).numpy())
  vis1 = np.hstack(to_rgb(x).numpy())
  # vis0 = np.hstack(x0.numpy())
  # vis1 = np.hstack(x.numpy())    
  vis = np.vstack([vis0, vis1])
  # imwrite('train_log/batches_%04d.jpg'%step_i, vis)
  if verbose:
      print('batch (before/after):')
  imshow(vis)

def plot_loss(loss_log):
  plt.figure(figsize=(10, 4))
  plt.title('Loss history (log10)')
  plt.plot(np.log10(loss_log), '.', alpha=0.1)
  plt.show()
    
def plot_losses(losses, labels=None, name=None):
    """Plot all the passed-in losses in a single figure
    
    :param losses: list, all the list of losses to be plotted
    :param labels: list, labels which to assign to respective losses
    :param name: str, if None, figure will not be saved, otherwise save to filename `name`
    """
    if labels is not None:
        assert len(losses) == len(labels)
        
    plt.figure(figsize=(10, 4))
    for i in range(len(losses)):
        if labels:
            plt.plot(np.log10(losses[i]), '.', alpha=0.1, label=labels[i])
        else:
            plt.plot(np.log10(losses[i]), '.', alpha=0.1)
    plt.ylim(top=-1)
    if labels:
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1) 
    plt.title(f"Comparison of log10 losses for {len(losses)} different models")
    plt.xlabel('Training steps')
    plt.ylabel('Log(10) loss')
    
    if name is not None:
        plt.savefig(name, format='png')
    plt.show()
        
    
# Defines class for making video demos of CA growth.
# Adapted from original implementation (not my own)
class VideoWriter:
  def __init__(self, filename, fps=50.0, **kw):
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
    
# Defines class for pooling figures (taken from original Distill paper)
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
      # setattr(self, k, np.asarray(v))
      setattr(self, k, v)        

  def sample(self, n):
    idx = np.random.choice(self._size, n, False)
    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)
    
## Additional utilities - these are largely my own

def clip_tensor(tensor):
    """Make `tensor` appropropriate for use with matplotlib.
        This involves detaching, moving to numpy, clipping to
        positive values and re-normalizing to [0...255.0].
    """
    return np.uint8(np.clip(tensor.detach().cpu().numpy(), 0, 1) * 255.0)

def simulate_model(model, init, n_steps, print_sim=True, device='cpu'):
    """Runs the simulation for ca model `models` for n_steps, starting
        from initial condition `init`.
    
    :param model: PyTorch model object
    :param init: PyTorch Tensor, starting state for model of shape either
        (B, H, W, C) or (H, W, C).
    :param print_sim: bool, whether or not to print resulting simulation
    :param device: device on which to run simulation
    :return: PyTorch tensor, result of simulating the model for `n_steps` steps.
    """
    with torch.no_grad():
        x, model = init.to(device), model.to(device)
        for _ in tqdm.trange(n_steps):
            x = model(x)
        if print_sim:
            if init.shape[0] == 1:
                # If batch size is 1, use matplotlib's imshow
                x0, x = clip_tensor(init), clip_tensor(x)
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(x0[0, ..., :4])
                ax2.imshow(x[0, ..., :4])
                fig.show()            
            else:
                visualize_batch(init.detach().cpu(), x.detach().cpu(), n_steps)
        return x

## Utilities for storing and loading objects

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
        return ca.to(device)
    else:
        ca.load_state_dict(torch.load("./models/{}.pth".format(model_name)))
        # ca.eval()
        return ca

def save_np_array(loss_log, filename, foldername="loss_log"):
    """Save loss_log array as filename in foldername folder
    
    :param loss_log: np.array, loss_log array
    :param filename: str, name under which to save loss_log array
    """
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    with open(f"./{foldername}/{filename}.npy", 'wb') as file:
        np.save(file, loss_log)
    print(f"Saved array under {foldername}/{filename} name to disk")
    
def load_np_array(filename, foldername="loss_log"):
    """Load `filename` from `foldername` folder in working directory
    
    :param filename: str, name of saved object
    :param foldername: str, name of folder from which to load `filename`
    :return: np.array loss_log array loaded from disk
    """
    with open(f"./{foldername}/{filename}.npy", 'rb') as file:
        return np.load(file, allow_pickle=True)

def get_stored_model(N, hidden_repr, hidden_repr_size, device='cpu'):
    """Calls the custom-made `create_filename` and `load_ca_model` 
        functions to return a loaded model object. Note: assumes that 
        the only differences in the models comes from the number of group 
        elements N in the cyclic group C_N, the hidden representation used, 
        and the number of internal feature fields.
        
        WARNING: Does not check if the requested parameters lead to a trained
        model. We leave that to the discretion of the user. 
        
    :param N: int, defines the 'N' in the cyclic group C_N
    :param hidden_repr: str, which hidden representation to use
    :param hidden_repr_size: int, defines how many internal feature fields to use
    :return: filename of stored model and trained PyTorch model object
    """
    filename = create_filename(N,
                                 hidden_repr=hidden_repr,
                                 hidden_repr_size=hidden_repr_size, # Note: this must be a multiple of channel_n
                                 channel_n=CHANNEL_N,
                                 hidden_size=HIDDEN_SIZE)

    loaded_model = load_ca_model(CAModel, filename, device=device,
                                 N=N,
                                 hidden_repr=hidden_repr,
                                 hidden_repr_size=hidden_repr_size,
                                 channel_n=CHANNEL_N,
                                 hidden_size=HIDDEN_SIZE)
    
    return filename, loaded_model.to(device)
    
    
#############################################################
# Helper functions to make a seed (either symmetric or asymmetric)
#############################################################

def get_circle_points(a, b, radius, num_points=3):
    """Returns three non-collinear points on a circle, centered
        at (a, b) with radius `radius`.
    
    :param a: int, x coordinate of circle center
    :param b: int, y coordinate of circle center
    :param radius: int, radius of circle 
    :param num_points: int, number of points for asymmetric seed
    """
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = a + radius * math.cos(angle)
        y = b + radius * math.sin(angle)
        points.append((int(round(x)), int(round(y))))
    return points

def make_seed(target_img, radius=None, channel_n=16, rot=0, p=16, device='cpu', print_seed=True):
    """Makes an asymmetric seed with three non-collinear points.
        If radius is None, seeds a single pixel.
        `rot` is an int in [1,2,3,4], representing 90 degree
        rotations to the left."""
    target_img = torch.tensor(target_img)
    pad_target = torch.nn.functional.pad(target_img, (0, 0, p, p, p, p))
    h, w = pad_target.shape[:2] # get height and width of padded target image
    x, y = h//2, w//2 # get coordinates of center pixel
    seed = torch.zeros(h, w, channel_n, dtype=torch.float32)
    if radius is None:
        seed[x, y, 3:] = 1.0    
    else:
        # Rotate the points
        points = get_circle_points(x, y, radius)
        points = [points[(i+rot) % len(points)] for i in range(len(points))]

        # Seed the points
        for color_channel_i, point in enumerate(points):
            seed[point[0], point[1], 3:] = 1.0  # set auxiliary channels
            seed[point[0], point[1], color_channel_i] = 1.0  # set color channel
        
    if print_seed:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        permuted_seed = seed.permute(-1, 0, 1)
        ax1.imshow(seed[..., :4])
        ax2.imshow(pad_target)
        
        ax1.set_title("Seed")
        ax2.set_title("Target")
        fig.show()
        
    return seed.to(device), pad_target.to(device)

#########################################################  


def make_video(models, seed, n_steps, video_name="test_video", device=None):
    """Makes a video using the VideoWriter class written by the original
        implementation's authors. Does not allow for intermediate
        transformations of the input (i.e. rotations).
        
    :param models: list, trained PyTorch models to be simulated
    :param seed: PyTorch tensor, initial seed to be shared by all models
    :param n_steps: int, number of model iterations to simulate
    """
    if '.mp4' not in video_name:
        video_name += '.mp4'
    x = torch.repeat_interleave(seed[None, :, :, :], len(models), dim=0).detach().cpu()
    with VideoWriter(video_name) as vid:
      for i in tqdm.trange(n_steps):
        vis = np.hstack(to_rgb(x))
        vid.add(zoom(vis, 2))
        for ca, xk in zip(models, x):
          ca_in = xk[None, ...].to(device) if device else xk[None, ...]
          xk[:] = ca(ca_in).detach().cpu()[0]

    # Make a VideoFileClip object and then write it 
    clip = mvp.VideoFileClip(video_name)
    clip.write_videofile(f'{video_name}')
    
def make_video_with_rotations(models, seed, n_steps, time_steps, angles, video_name="original_rot.mp4", device=None):
    """Similar to `make_video` function, except allows the user too pass in a list of time_steps 
        and a list of angles to apply rotations to the model. Should be used only for the original implementation.
        
        Note: len(angles) must equal len(time_steps)
        
    :param models: list, trained PyTorch models to be simulated
    :param seed: PyTorch tensor, initial seed to be shared by all models
    :param n_steps: int, number of model iterations to simulate
    :param time_steps: list, time steps at which to apply an angle from `angles` list
    :param angles: list, angles of rotation to apply to the model.
    """
    assert len(time_steps) == len(angles)
    if '.mp4' not in video_name:
        video_name += '.mp4'
    x = torch.repeat_interleave(seed[None, :, :, :], len(models), dim=0).detach().cpu()
    angle = 0.0
    angle_formula = lambda a : a/360.0 * 2 * np.pi
    with VideoWriter(video_name) as vid:
      for i in tqdm.trange(n_steps):
        vis = np.hstack(to_rgb(x))
        vid.add(zoom(vis, 2))
        for ca, xk in zip(models, x):
          ca_in = xk[None, ...].to(device)
          for idx, time_step in enumerate(time_steps):
            if time_step == 0:
              angle = angle_formula(angles[idx])
            elif i % time_step == 0 and i != 0:
              angle = angle_formula(angles[idx])
          xk[:] = ca(ca_in, angle=angle).detach().cpu()[0]

    # Make a VideoFileClip object and then write it 
    clip = mvp.VideoFileClip(video_name)
    clip.write_videofile(f'{video_name}')       
    
    
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