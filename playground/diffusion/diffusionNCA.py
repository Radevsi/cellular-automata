import os
import io
import PIL.Image, PIL.ImageDraw
import matplotlib.pyplot as plt
import requests
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import time

# Custom helpers package
from helpers import *

# For notebook
from IPython.display import clear_output

# For reloading the package
import importlib
import sys
importlib.reload(sys.modules['helpers'])
from helpers import *

print("Finished imports from file")

# Set cuda gpu
device_id = 0
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')


CHANNEL_N = 16
HIDDEN_SIZE = 128
PERCEPTION_SIZE = 48  # size of perception vector
BATCH_SIZE = 8
CELL_FIRE_RATE = 0.5
TARGET_PADDING = 16


class NCADiff(nn.Module):
    def __init__(self, c_in, c_noise=3,
                 hidden_size=HIDDEN_SIZE, perception_size=PERCEPTION_SIZE,
                 time_dim=256
                ):
        
        super().__init__()
        
        assert c_in % c_noise == 0
        
        self.c_in = c_in
        self.c_noise = c_noise
        self.hidden_size = hidden_size
        self.perception_size = perception_size
        self.time_dim = time_dim
        
        self.perceive = nn.Conv2d(in_channels=self.c_in,
                                  out_channels=self.perception_size,
                                  kernel_size=3,
                                  padding='same',
                                  groups=self.c_in,
                                 )
        
        ## CA Rule
        conv1 = nn.Conv2d(in_channels=self.perception_size, out_channels=self.hidden_size, kernel_size=1)
        # conv2 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=1)
        conv2 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.c_in, kernel_size=1, bias=False)

        nn.init.zeros_(conv2.weight)
        
        self.rule = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
        )        
        
        self.noise_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_noise, kernel_size=1)
        
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            # Again, shrink down to noised channels (just rgb for now)
            nn.Linear(in_features=self.time_dim, out_features=self.c_in), 
        )     
                          
    def pos_encoding(self, t, channels):
        """Encode time tensor using sinusoidal embedding"""
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc                          
                          
    def forward(self, x, t):
        """Assume PyTorch shaping convention. Take in noised image and time,
            Use time embedding, return a "denoised image" and the noise
            prediction
        """
        y = self.perceive(x)
        dx = self.rule(y)
        x = x + dx
        
        noise_channel = self.noise_conv(x)
        
        # Add time embedding to channel noise
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.time_dim)        
        embedding = self.time_embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        # Add time embedding to new x
        x = x + embedding
        
        return x, noise_channel
    

#################### DIFFUSION #####################

class Diffusion:
    def __init__(self, c_in, c_noise, noise_steps=1000, beta_start=1e-4, beta_end=0.2, img_size=28):
        self.c_in = c_in
        self.c_noise = c_noise
        self.noise_steps = noise_steps
        self.beta_start, self.beta_end = beta_start, beta_end
        self.img_size = img_size
        
        # Noise scheduler
        self.beta = self.prepare_noise_scheduler().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
      
    def prepare_noise_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x) # random noise

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps    
    
    def sample_timesteps(self, n):
        """Define how many times to denoise the model, `n` being the number
            of timesteps
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        
        logging.info(f'Sampling {n} new images...')
        print(f"Sampling {n} images")
        model.eval()
        
        with torch.no_grad():            
            
            x = torch.randn((n, self.c_noise, self.img_size, self.img_size)).to(device)   
            model_input = x.tile((1, self.c_in//self.c_noise, 1, 1))
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                
                # Create a tensor of length `n` to create the timesteps
                t = (torch.ones(n) * i).long().to(device)
    
                # iter_n = np.random.randint(30, 40, dtype=np.int32)
                # model_input[:, :self.c_noise, :, :] = x
                model_input = x.tile((1, self.c_in//self.c_noise, 1, 1))
                for _ in range(1):
                    model_input, predicted_noise = model(model_input, t)
                # x = model_input[:, :self.c_noise, :, :]
                
                # For noise scheduling
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]     

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.randn_like(x)                

                # Perform one denoising step
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise                   
                
                
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x                
                
            
############################# TRAIN ################################


class TrainDiff():
    def __init__(self, model, diffusion, dataloader, c_in, run_name, c_noise=3, lr=3e-4):
        """Training class.
            `c_noise` defines number of noise channels. Default to 3.
        """
        
        setup_logging(run_name)
        
        self.model = model
        self.diffusion = diffusion

        # self.logger = SummaryWriter(os.path.join('runs', run_name))
        self.dataloader = dataloader
        self.run_name = run_name
        self.c_in = c_in
        self.c_noise = c_noise
        
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss()
    
        self.losses = []
        
        self.l = len(dataloader)                
        
    def train(self, epochs):
        
        for epoch in range(epochs):
            logging.info(f'Starting epoch {epoch}:')
            print(f'Starting epoch {epoch}:')
            pbar = tqdm(self.dataloader)
            
            for i, (images, _) in enumerate(pbar):
                images = images.to(device)
                
                t = self.diffusion.sample_timesteps(images.shape[0]).to(device)
                x, noise = self.diffusion.noise_images(images, t)
                
                # Tile the noise to the hidden channels
                model_input = torch.tile(x, (1, self.c_in // self.c_noise, 1, 1))        

                noise_channel, loss = self.train_step(model_input, t, images, noise)
                self.losses.append(loss)
                
                pbar.set_postfix(MSE=loss)
                
            if epoch % 20 == 0:
                sampled_images = self.diffusion.sample(self.model, n=BATCH_SIZE)
                save_images(sampled_images, os.path.join('results', self.run_name, f'{epoch}.png'))
            
            if epoch % 5 == 0:
                save_np_array(self.losses, f"{self.run_name}")
                torch.save(self.model.state_dict(), os.path.join("models", self.run_name, f'ckpt{epoch}.pt'))            
        return self.losses
            
            
    def train_step(self, x, t, images, noise):
        
        iter_n = np.random.randint(30, 40, dtype=np.int32)
        for _ in range(iter_n):
            # Forward pass of model
            x, noise_channel = self.model(x, t)
            
        loss = self.mse(noise, noise_channel)
            
        # Compute gradients
        loss.backward()

        # Apply L2 normalization to parameter gradients as per original paper
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = p.grad / (p.grad.norm() + 1e-8) 

        # Update parameters
        self.optim.step()

        # Clear previous gradients accumulated on parameters
        self.optim.zero_grad()
        
        noise_channel = (noise_channel.clamp(-1, 1) + 1) / 2
        noise_channel = (noise_channel * 255).type(torch.uint8)          

        return noise_channel, loss.item()
                
        
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

        
def get_cifar(download=False, img_size=28, batch_size=8):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    K = 5000
    
    dataset_cifar = torchvision.datasets.CIFAR10(root='datasets', train=False, download=download, transform=transforms)
    subsample_train_indices = torch.randperm(len(dataset_cifar))[:K]
    subloader_cifar = torch.utils.data.DataLoader(dataset_cifar, batch_size=batch_size, 
                              sampler=torch.utils.data.SubsetRandomSampler(subsample_train_indices))     
    return subloader_cifar


def get_dtd():
    img_size, batch_size = 28, 8

    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) # global mean and std of MNIST
    ])
    K = 1000 # enter your length here
    dataset_dtd = torchvision.datasets.DTD(root='datasets', download=False, transform=transforms)
    
    
    subsample_train_indices = torch.randperm(len(dataset_dtd))[:K]
    subloader_dtd = torch.utils.data.DataLoader(dataset_dtd, batch_size=8, 
                              sampler=torch.utils.data.SubsetRandomSampler(subsample_train_indices))    
    
    return subloader_dtd  

    
if __name__ == "__main__":
    
################################## Model Architecture

    # c_noise = 3
    # channel_n = 120
    # perception_size = 720
    # hidden_size = 1024

    c_noise = 3
    channel_n = 16
    perception_size = 48
    hidden_size = 128

    # c_noise = 1
    # channel_n = 90
    # perception_size = 360
    # hidden_size = 1024


################################# Dataset
    
    # loaders = torch.load('mnist_classed.pkl')
    # dataloader_0 = loaders[0] # MNIST for the 5 class
    # subloader_mnist = get_dataloader()
    
    
    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/dotted/dotted_0201.jpg'
    polkadots_img = imread(url, max_size=64)
    
    # Convert and transform
    img_size, batch_size = 64, 8
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) # global mean and std of MNIST
    ])
    target_img = transforms(np2pil(polkadots_img))

    # Make dataloader from target
    dataloader_polkadots = make_dataloader(target_img, n_images=200)  


    # dataloader_cifar = get_cifar(download=False)

    # dataset_dtd = get_dtd()

######################## Model Instantiations

    ca_diff = NCADiff(c_in=channel_n, c_noise=c_noise, hidden_size=hidden_size, 
                     perception_size=perception_size).to(device)

    diffusion = Diffusion(c_in=channel_n, c_noise=c_noise, noise_steps=1000, img_size=64)

    
    start_time = time.time() # time training run    
    
    run_name = 'NCADiff_small_polkadots'
    train_diff = TrainDiff(ca_diff, diffusion, dataloader=dataloader_polkadots, c_in=channel_n,
                           c_noise=c_noise, run_name=run_name)

    losses = train_diff.train(epochs=201)            
    
    elapsed_time = np.round((time.time() - start_time) / 60, decimals=4)
    print(f'\nElapsed time for training run for {run_name}: {elapsed_time}')
            
    