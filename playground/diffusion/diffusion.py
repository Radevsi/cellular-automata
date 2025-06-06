import torch
import logging
from tqdm import tqdm


class Diffusion:
  def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cpu'):
    if device == 'cpu':
        print("WARNING: Diffusion model is on cpu")
    self.noise_steps = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.img_size = img_size
    self.device = device

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
    return torch.randint(low=1, high=self.noise_steps, size=(n,))

  def sample(self, model, n):
    """Samples `n` images using trained `model`"""
    
    logging.info(f'Sampling {n} new images....')
    model.eval()
    with torch.no_grad():

      # Create initial images by sampling from the Normal distribution
      x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

      # Go through all noise steps in reverse order 
      for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

        # Create a tensor of length n to create the timesteps
        t = (torch.ones(n) * i).long().to(self.device)
        predicted_noise = model(x, t)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        # We don't want to add noise in the last iteration
        if i > 1:
          noise = torch.randn_like(x)
        else:
          noise = torch.zeros_like(x)

        # Finally, alter the images with a little bit of noise according to Alg 2
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) +\
              torch.sqrt(beta) * noise

    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x