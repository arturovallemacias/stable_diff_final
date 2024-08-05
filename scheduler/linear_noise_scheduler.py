
import torch
import numpy as np 

class LinearNoiseScheduler: 
      def __init__(self, num_timesteps,beta_start, beta_end): 
          self.num_timesteps = num_timesteps 
          self.beta_start = beta_start
          self.beta_end = beta_end 
          
          self.betas = (
              torch.linspace(beta_start**0.5, beta_end** 0.5, num_timesteps) **2 
          )
      
          self.alphas = 1. - self.betas
          self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0) 
          self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod) 
 
          self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1-self.alpha_cum_prod) 
     
      def add_noise(self, original, noise, t):

          original_shape = original.shape 
          batch_size = original_shape[0]   

