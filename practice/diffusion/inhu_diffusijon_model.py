from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
from torch import Tensor

class InhuDiffusionModel(nn.Module):
    def __init__(self, coding_dim:int, dropout:float=0.1, total_timestep:int=4000, time_dim:int=256):
        super().__init__()
        self.s:float = 0.008
        self.b_max:float = 0.999
        self.total_timestep = total_timestep
        
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
    #noise_image = [b, rgb, w, h], time_steps[b, image, time_steps]
    def forward(self, Noise_image:Tensor, time_steps:Tensor):
        embeded_time = self.time_mlp(self.sinusoidal_embedding(time_steps)) #[B, time_dim]
        predicted_original_image:Tensor = self.get_original_image()
    
    def sinusoidal_embedding(self, time_steps: Tensor) -> Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(-torch.arange(half, device=time_steps.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = time_steps[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    #predicted_noise = [b, rgb, w, h], noised_image = same, a_t_bar = [b,image,time_steps]
    def get_original_image(self, predicted_noise:Tensor, noised_image:Tensor, a_t_bar:Tensor) -> Tensor:
        original_image:Tensor = noised_image - torch.sqrt(1 - a_t_bar) * predicted_noise / torch.sqrt(a_t_bar)
        return original_image
        
    #generate noise image --------------------------------------
    def get_noised_image(self, target_time_steps:Tensor, original_image:Tensor, noise:Tensor, a_t_bar:Tensor) -> Tensor:
        a:Tensor = a_t_bar
        results:Tensor = original_image * torch.sqrt(a) + (torch.sqrt(1 - a) * noise)
        return results
    
    def cos_func(self, time_steps:Tensor) -> Tensor:
        nominator = (time_steps / self.total_timestep + self.s) * torch.pi
        denominator = (1 + self.s) * 2
        return torch.cos(nominator / denominator) ** 2
    
    def get_a_t_bar(self, time_steps:Tensor) -> Tensor:
        return self.cos_func(time_steps) / self.cos_func(torch.zeros_like(time_steps))
    
    def get_b_t(self, time_steps:Tensor, a_t_bar:Tensor, a_t_bar_past:Tensor) -> Tensor:
        return 1 - (a_t_bar / a_t_bar_past)

class InhuResBlock(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, time_dim:int=256, dropout:float=0.1, num_groups:int=8):
        super().__init__()
        if input_dim % num_groups != 0:
            raise ValueError(f"input_dim:{input_dim} % norm_groups:{num_groups} != 0")
        
        if output_dim % num_groups != 0:
            raise ValueError(f"input_dim:{output_dim} % norm_groups:{num_groups} != 0")
        self.norm1 = nn.GroupNorm(num_groups, input_dim)
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, output_dim * 2)
        )
        
        self.norm2 = nn.GroupNorm(num_groups, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        
        if input_dim == output_dim:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Conv2d(input_dim, output_dim, kernel_size=4)
        
    def forward(self, input:Tensor, embeded_time:Tensor):
        h1 = self.norm1(input)
        h1 = nn.GELU(h1)
        h1 = self.conv1(h1) #[batch, output_dim, h, w]
        
        scale, shift = self.time_proj(embeded_time).chunk(2, dim=1) # each of them is [batch, outputdim]
        
        scale, shift = scale[:, :, None, None], shift[:, :, None, None]
        
        h2 = self.norm2(h1)
        h2 = h2 * (1 + scale) + shift
        h2 = nn.GELU(h1)
        h2 = self.dropout(h2)
        h2 = self.conv2(h2)
        
        return h2 + self.residual_proj(input)
        
        
        
        