import os
import torch
import torchvision
import torch.nn as nn
from torch import optim
import logging
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from unet import UNet
from diffusion import Diffusion

from utils import get_data, setup_logging, save_images, imread, np2pil, save_np_array, load_np_array


class Train():
  def __init__(self, model, diffusion, dataloader, run_name, lr=3e-4):
    """Training class"""
    
    setup_logging(run_name)
    self.model = model
    self.diffusion = diffusion
    
    # self.logger = SummaryWriter(os.path.join('runs', run_name))
    self.dataloader = dataloader
    self.run_name = run_name

    self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
    self.mse = nn.MSELoss()

    self.losses = []
    
    self.l = len(dataloader)

  def train(self, epochs):
    for epoch in range(epochs):
      logging.info(f"Starting epoch {epoch}:")
      print(f"Starting epoch {epoch}:")
        
      pbar = tqdm(self.dataloader)
      
      for i, (images, _) in enumerate(pbar):
        images = images.to(device)
        t = self.diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = self.diffusion.noise_images(images, t)
        predicted_noise = self.model(x_t, t)
        loss = self.mse(noise, predicted_noise)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        
        pbar.set_postfix(MSE=loss.item())
        
        # self.logger.add_scalar('MSE', loss.item(), global_step=epoch * self.l + i)

      if epoch % 10 == 0:
          sampled_images = self.diffusion.sample(self.model, n=images.shape[0])
          save_images(sampled_images, os.path.join("results", self.run_name, f'{epoch}.jpg'))
      
      if epoch % 5 == 0:
          save_np_array(self.losses, f'{self.run_name}')
          torch.save(self.model.state_dict(), os.path.join("models", self.run_name, f'ckpt.pt'))

    
    
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
    
    
# For singly imported images
def make_dataloader(target_img, n_images=1, batch_size=8):
    """
        Note: Expects pad_target to have shape (4, 64, 64)
    """
    batched_data = torch.repeat_interleave(target_img[None, ...], batch_size, dim=0) 
    return [(batched_data, 'label')] * n_images    
    
    
if __name__ == "__main__":
    
    device_id = 2
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
    
    
    url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/dotted/dotted_0201.jpg'
    polkadots_img = imread(url, max_size=64)
    
    # Convert and transform
    img_size, batch_size = 64, 8
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)) # global mean and std of MNIST
    ])
    target_img = transforms(np2pil(polkadots_img))

    # Make dataloader from target
    dataloader_polkadots = make_dataloader(target_img, n_images=50)  
    
    
    # dataloader_cifar = get_cifar(download=False)
    
    start_time = time.time() # time training run    

    
    # Begin training
    model = UNet(device=device).to(device)
    diffusion = Diffusion(device=device, img_size=64, noise_steps=500)

    # data_loader = get_data('datasets', 'cifar')
    
    train = Train(model, diffusion, dataloader_polkadots, "UNet500_polka")
    train.train(epochs=151)    

    elapsed_time = np.round((time.time() - start_time) / 60, decimals=4)
    print(f'\nElapsed time for training run for {run_name}: {elapsed_time}')    
    