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

# from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

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
device_id = 1
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')


CHANNEL_N = 16
HIDDEN_SIZE = 128
PERCEPTION_SIZE = 48  # size of perception vector
BATCH_SIZE = 8
CELL_FIRE_RATE = 0.5
TARGET_PADDING = 16


class CANet(nn.Module):
    def __init__(self, c_in, c_noise=3, hidden_size=HIDDEN_SIZE, perception_size=PERCEPTION_SIZE,
                 time_dim=256, fire_rate=CELL_FIRE_RATE):
        """Note: c_in must be passed in. This include both rgb and hidden channels. 
            `c_noise` defines how many channels should be noised.
        
            `time_dim` parameter unchanged from original diffusion implementation.
            
            `fire_rate` is for stochastic updates (same as original CA design)
        """
        super().__init__()
        
        assert c_in % c_noise == 0 # This must hold for my sanity preservation
        
        self.c_in = c_in
        self.c_noise = c_noise
        
        # Number of channels total
        # self.channels = c_in / c_noise if c_in > c_noise else c_noise
        
        self.hidden_size = hidden_size
        self.time_dim = time_dim
        
        self.fire_rate = fire_rate
        
        # Can either make learnable perception filters or hardcode (look at texture paper)
        self.perceive = nn.Conv2d(in_channels=c_in, out_channels=perception_size, kernel_size=3, padding='same', groups=c_in)
        
        ## CA Rule
        conv1 = nn.Conv2d(in_channels=perception_size, out_channels=hidden_size, kernel_size=1)
        
        # out_channels should shrink down to noised channels
        conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=c_in, kernel_size=1) 
        
        # Add another layer to bring down to noise channels
        conv3 = nn.Conv2d(in_channels=c_in, out_channels=c_noise, kernel_size=1, bias=False)
        
        # Apply "do-nothing" initial behavior - not sure about this here
        torch.nn.init.zeros_(conv3.weight)
        # torch.nn.init.zeros_(conv1.bias)
        # torch.nn.init.zeros_(conv2.bias)   
        
        self.rule = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3
        )        
        
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=self.time_dim, out_features=3), # Again, shrink down to noised channels (just rgb for now)
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
    
    def forward(self, x, t, fire_rate=None):
        """Assumes `x` has shape [B, C, H, W]
            Must also take in `t` and embed time dimension.
            
            Note: this `x` will have the same shape as the input
            image. If we want hidden channels, we will add them here. 
        """
        
        # First let's create hidden channels. Can initialize with ones or random
        if self.c_in > self.c_noise:
            # c_in must be a multiple of c_noise
            x = torch.tile(x, (1, self.c_in // self.c_noise, 1, 1))
        
        # Neighbourhood information
        perception = self.perceive(x)
        
        # Note that this should bring you down to noise channels
        rule_output = self.rule(perception) 
        
        # Time embedding
        t = t.unsqueeze(-1).type(torch.float32)
        t = self.pos_encoding(t, self.time_dim)
        embedding = self.time_embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        # Add time embedding to rgb channels (indexing should be unnecessary)
        rule_output[:, :self.c_noise, :, :] = rule_output[:, :self.c_noise, :, :] + embedding
        
#         # Update mask for stochasticity (may be optional...)
#         if fire_rate is None:
#             fire_rate = self.fire_rate
#         update_mask = torch.rand(x[:, :1, :, :].shape, device=device, dtype=torch.float32) <= fire_rate
#         rule_output = rule_output * update_mask
        
        return rule_output # Assume this is noise
    
    
#################### DIFFUSION ###########################


class DiffusionCA:
    def __init__(self, c_in, c_noise=3, noise_steps=1000, beta_start=1e-4, beta_end=0.2, img_size=64):
        """Note: `c_in` defines number of input channels. Must be provided.
        
            `c_noise` determines number of channels to be noised. Must be less than `c_in`
        """
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
        """Sample `n` images using `model` to remove the noise.
        
            Note: Need to standardize how to handle the number of channels
        """
        logging.info(f"Sampling {n} new images...")
        model.eval()
        
        # Make seed (set hidden channels to 1 if c_in > 3)
        # seed = torch.zeros(n, self.c_in, self.img_size, self.img_size).to(device)
        # if self.c_in > self.c_noise: # this ignores alphas for now
        #     seed[:, self.c_noise:, self.img_size//2, self.img_size//2] = 1.0
        
        with torch.no_grad():
            
            # Create initial "images" by sampling from Gaussian - default to just 3 channels
            x_T = torch.randn((n, self.c_noise, self.img_size, self.img_size)).to(device)
            x = x_T
            
            
            # Augment with hidden channels (if c_in > 3)
            # seed[:, :self.c_noise, :, :] = x_T
            # x = seed # rename
            
            # Iterate through noise steps in reverse order and denoise
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                
                # Create a tensor of length `n` to create the timesteps
                t = (torch.ones(n) * i).long().to(device)
                
                # Add hidden channels to the input of the model
                
                # Here is where we need to decide how to handle the CA iteration
                # For now, assume the network still outputs noise removal updates
                iter_n = np.random.randint(64, 97, dtype=np.int32)
                for _ in range(1):
                    predicted_noise = model(x, t)

                    # For noise scheduling
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]     

                    if i > 1:
                        noise = torch.randn_like(x_T)
                    else:
                        noise = torch.randn_like(x_T)

                    # Denoise the image for one step
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) +\
                              torch.sqrt(beta) * noise   
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
                
        
######################## TRAINING #################################

# Training loop

class Train():
    def __init__(self, model, diffusion, dataloader, run_name, c_noise=3, lr=3e-4):
        """Training class.
            `c_noise` defines number of noise channels. Default to 3.
        """
        
        setup_logging(run_name)
        
        self.model = model
        self.diffusion = diffusion

        # self.logger = SummaryWriter(os.path.join('runs', run_name))
        self.dataloader = dataloader
        self.run_name = run_name
        self.c_noise = c_noise
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss()

        self.l = len(dataloader)
        
    def train(self, epochs):
        
        for epoch in range(epochs):
            logging.info(f'Starting epoch {epoch}:')
            pbar = tqdm(self.dataloader)
            
            for i, (images, _) in enumerate(pbar):
            # for i, images in enumerate(pbar):
                images = images.to(device)
                t = self.diffusion.sample_timesteps(images.shape[0]).to(device)
                
                # Add noise to the image
                # if self.c_noise > images.shape[1]:
                    
                
                images = images[:, :self.c_noise, :, :]
                x_t, noise = self.diffusion.noise_images(images, t)
                
                iter_n = np.random.randint(64, 97, dtype=np.int32)                
                for _ in range(iter_n):

                    predicted_noise = self.model(x_t, t)
                    loss = self.mse(noise, predicted_noise)

                    # Backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                pbar.set_postfix(MSE=loss.item())
                
                # self.logger.add_scalar("MSE", loss.item(), global_step=epoch * self.l + i)
                
            # Sample images
            # clear_output()
            sampled_images = self.diffusion.sample(self.model, n=images.shape[0])
            # plot_images(sampled_images)
            
            
            save_images(sampled_images, os.path.join('results', self.run_name, f'{epoch}.png'))
            save_np_array(self.losses, f"{self.run_name}")
            torch.save(self.model.state_dict(), os.path.join("models", self.run_name, f'ckpt.pt'))
            
    
    
# Pull datasets (if using emoji, still organize as dataloader

def make_dataloader(target_img, n_images=1, batch_size=8):
    """
        Note: Expects pad_target to have shape (4, 64, 64)
    """
    batched_data = torch.repeat_interleave(target_img[None, ...], batch_size, dim=0) 
    return [(batched_data, 'label')] * n_images

    
if __name__ == "__main__":
    
    c_noise = 3
    c_in = 90
    p_size = 360 # 90 * 4
    h_size = 1024

    # c_noise = 3
    # c_in = 3
    # p_size = 15 
    # h_size = 128

    ca = CANet(c_in=c_in, c_noise=c_noise, perception_size=p_size, hidden_size=h_size).to(device)
    diffusion = DiffusionCA(c_in=c_in, c_noise=c_noise, noise_steps=1000)

    print(f"Starting training with {c_noise} noised channels, {c_in} input channels, {p_size} perception vector size, and {h_size} hidden size")
    

    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/dotted/dotted_0201.jpg'
    polkadots_img = imread(url, max_size=64)
    
    # Convert and transform
    img_size, batch_size = 64, 8
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) # global mean and std of MNIST
    ])
    target_img = transforms(np2pil(polkadots_img))

    # Make dataloader from target
    dataloader = make_dataloader(target_img, n_images=200)    

    train = Train(ca, diffusion, dataloader, run_name="polkadots_all_noised", c_noise=c_noise)
    train.train(epochs=100)