import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'

import io
import PIL.Image, PIL.ImageDraw

import requests
import logging

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Notebook dependencies
from IPython.display import clear_output, Image

from helpers import *

import importlib
import sys
importlib.reload(sys.modules['helpers'])
from helpers import *

device_id = 0
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
print('device is {}'.format(device))


CHANNEL_N = 16
HIDDEN_SIZE = 128
PERCEPTION_SIZE = 48
BATCH_SIZE = 8


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


# Define Model

class NCAModel(nn.Module):
  def __init__(self, 
               channel_n=CHANNEL_N, 
               hidden_size=HIDDEN_SIZE,
               perception_size=PERCEPTION_SIZE
              ):
    super().__init__()
    
    self.channel_n = channel_n
    self.hidden_size = hidden_size
    self.perception_size = perception_size

    
    self.perceive = nn.Conv2d(in_channels=self.channel_n, 
                              out_channels=perception_size, kernel_size=3, padding='same', groups=self.channel_n)
        
    conv1 = nn.Conv2d(in_channels=perception_size, out_channels=self.hidden_size, kernel_size=1)
    conv2 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.channel_n, kernel_size=1, bias=False)
    
    self.conv_out = nn.Conv2d(in_channels=self.channel_n, out_channels=1, kernel_size=1)
    
    # Apply "do-nothing" initial behavior
    torch.nn.init.zeros_(conv2.weight)

    self.rule = nn.Sequential(
        conv1,
        nn.ReLU(),
        conv2,
    )
    
  def forward(self, x):
    """Assume PyTorch shaping convention this time"""

    y = self.perceive(x) # y should be on device
    dx = self.rule(y)
    x = x + dx
    
    single_channel = self.conv_out(x)
    
    return x, single_channel

###################### Training Sequence ###########################

class TrainNCA:
  def __init__(self, ca_model, seed, dataloader, channel_n, run_name, lr=2e-3):
    """Initialize training object with model to train, seed and target
    
    :param ca_model: PyTorch model class object (instantiated)
    :param seed: PyTorch tensor of shape (H, W, C), initial seed to start training
    :param dataloader: Torchvision dataloader object
    :param channel_n: Number of channels used in model
    """
    setup_logging(run_name)
    
    self.ca = ca_model
    self.seed = seed
    self.dataloader = dataloader
    self.channel_n = channel_n
    self.run_name = run_name
    
    # Initialize optimizer with lr scheduler
    self.optim = torch.optim.Adam(self.ca.parameters(), lr=lr)
    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2000], gamma=0.1)
    
    # Use library MSELoss
    self.mse = nn.MSELoss()
    
    # Log the training progress
    self.loss_log = []

  def train(self, epochs):
    """Train with epochs this time around because we deal
        with dataloader
    """
    
    for epoch in range(epochs):
        logging.info(f'Starting epoch {epoch}:')

        pbar = tqdm(self.dataloader)
        
        for i, (images, _) in enumerate(pbar):
        
            images = images.to(device)
        
            if self.seed is None:
                # Seed with noise
                x0 = torch.randn((images.shape[0], self.channel_n, images.shape[2], images.shape[3])).to(device)
            else:
                x0 = torch.repeat_interleave(self.seed[None, ...], BATCH_SIZE, 0)

            # Run through a single training step of the model
            single_channel, loss = self._train_step(x0, images)
            self.loss_log.append(loss)
            
            pbar.set_postfix(MSE=loss)

            
        # Export model
        clear_output()
        
        # Plot loss
        # plot_loss(self.loss_log)        
        
        save_images(single_channel, os.path.join('results', self.run_name, f'{epoch}.png'))
        save_np_array(self.loss_log, f"{self.run_name}")
        torch.save(self.ca.state_dict(), os.path.join("models", self.run_name, f'ckpt.pt'))

    return self.loss_log
    
  def _train_step(self, x, images):
    """Perform the update step some random number of times"""

    iter_n = np.random.randint(64, 97, dtype=np.int32)
    for _ in range(iter_n):
      # Forward pass of model
      x, single_channel = self.ca(x)

    # Compute loss (note we must take the mean across the batch dimension)
    loss = self.mse(images, single_channel).mean()

    # Compute gradients
    loss.backward()
    
    # Apply L2 normalization to parameter gradients as per original paper
    for p in self.ca.parameters():
      if p.grad is not None:
        p.grad = p.grad / (p.grad.norm() + 1e-8) 
        
    # Update parameters
    self.optim.step()
    
    # Clear previous gradients accumulated on parameters
    self.optim.zero_grad()

    # Update learning rate step
    self.scheduler.step()

    single_channel = (single_channel.clamp(-1, 1) + 1) / 2
    single_channel = (single_channel * 255).type(torch.uint8)   

    return single_channel, loss.item()

    
def get_dataloader():
    img_size, batch_size = 28, BATCH_SIZE

    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) # global mean and std of MNIST
    ])

    K = 1000 # enter your length here
    dataset_mnist = torchvision.datasets.MNIST(root='datasets', train=False, download=False, transform=transforms)
    subsample_train_indices = torch.randperm(len(dataset_mnist))[:K]
    subloader_mnist = torch.utils.data.DataLoader(dataset_mnist, batch_size=BATCH_SIZE, 
                              sampler=torch.utils.data.SubsetRandomSampler(subsample_train_indices))    
    
    return subloader_mnist  

    
# For singly imported images
def make_dataloader(target_img, n_images=1, batch_size=8):
    """
        Note: Expects pad_target to have shape (4, 64, 64)
    """
    batched_data = torch.repeat_interleave(target_img[None, ...], batch_size, dim=0) 
    return [(batched_data, 'label')] * n_images
    
    
    
if __name__ == "__main__":
    
    # Original CA Model

    
#     channel_n = 16
#     perception_size = 48
#     hidden_size = 128    
    
    channel_n = 90
    perception_size = 360
    hidden_size = 1024

    
    # loaders = torch.load('mnist_classed.pkl')
    # dataloader_5 = loaders[5] # MNIST for the 0 class
    
    subloader_mnist = get_dataloader()





#     url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/dotted/dotted_0201.jpg'
#     polkadots_img = imread(url, max_size=28)
    
#     # Convert and transform
#     img_size, batch_size = 28, 8
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5,), (0.5,)) # global mean and std of MNIST
#     ])
#     target_img = transforms(np2pil(polkadots_img))

#     # Make dataloader from target
#     dataloader_polkadots = make_dataloader(target_img, n_images=200)  

    
    
    
    nca_model = NCAModel(channel_n=channel_n, hidden_size=hidden_size, 
                         perception_size=perception_size).to(device)

    
    
    start_time = time.time() # time training run    
    run_name = 'NCA_big_MNIST_all'
    
    train_nca = TrainNCA(nca_model, seed=None, dataloader=subloader_mnist, 
                         channel_n=channel_n, run_name=run_name)    

    losses = train_nca.train(epochs=201)          
    
    elapsed_time = np.round((time.time() - start_time) / 60, decimals=4)
    print(f'\nElapsed time for training run for {run_name}: {elapsed_time}')    
    