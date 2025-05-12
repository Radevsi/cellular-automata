import os
import torch
import torchvision

import matplotlib.pyplot as plt
import PIL
import requests
import io
import numpy as np
from PIL import Image

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1)
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    
def get_data(dataset_path, dataset_name, img_size=64, batch_size=8):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'cifar' or dataset_name == 'cifar10' or dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='datasets', download=False, transform=transforms)
    elif dataset_name == 'mnist' or dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='datasets', download=False, transform=transforms)
    elif dataset_name == 'celeba' or dataset_name == 'celebA' or dataset_name == 'CelebA':
        dataset = torchvision.datasets.CelebA(root='datasets', download=False, transform=transforms)
    else:
        raise ValueError(f'dataset_name {dataset_name} is not supported')
        
    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size, shuffle=True)        
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
    
    
def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)    
    
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

