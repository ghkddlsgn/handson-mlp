import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
from torch import Tensor
from practice.diffusion.inhu_res_unet import InhuResUnet
from practice.diffusion.image_noise_generator import get_a_t_bar
import math

from typing import Any, TypedDict
class InhuDiffusionCheckpoint(TypedDict):
    epoch: int
    best_epoch: int
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    best_val_loss: float
    history: dict[str, list[float]]
    total_timestep: int
    time_dim: int


class InhuDiffusionModel(nn.Module):
    def __init__(self, unet:InhuResUnet, time_dim:int=256, total_timestep:int=4000):
        super().__init__()

        self.total_timestep = total_timestep
        
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        self.unet:InhuResUnet = unet
        
    #noise_image = [b, rgb, w, h], time_steps[b]
    def forward(self, noised_image:Tensor, time_steps:Tensor):
        embeded_time = self.time_mlp(self.sinusoidal_embedding(time_steps)) #[B]
        pred_noise = self.unet(noised_image, embeded_time)
        a_t_bar = get_a_t_bar(time_steps, self.total_timestep)
        pred_original_image:Tensor = self.get_original_image(pred_noise, noised_image, a_t_bar)
        
        return pred_noise, pred_original_image
            
    def sinusoidal_embedding(self, time_steps: Tensor) -> Tensor: #[b, time_embed]
        half = self.time_dim // 2
        freqs = torch.exp(-torch.arange(half, device=time_steps.device) * math.log(10000.0) / (half - 1))
        args = time_steps[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    #predicted_noise = [b, rgb, w, h], noised_image = same, a_t_bar = [b]
    def get_original_image(self, predicted_noise:Tensor, noised_image:Tensor, a_t_bar:Tensor)-> Tensor:
        a_t_bar = a_t_bar.to(device=noised_image.device, dtype=noised_image.dtype)
        a_t_bar = a_t_bar[:, None, None, None]
        original_image:Tensor = (noised_image - torch.sqrt(1 - a_t_bar) * predicted_noise) / torch.sqrt(a_t_bar)
        return original_image

    def generate_image(self, image_size:list[int], steps:int = 40, seed:int=42) -> Tensor:
        torch.manual_seed(seed)
        device = next(self.parameters()).device
        time_steps = torch.linspace(self.total_timestep - 1, 0, steps, dtype=torch.long, device=device)
        
        current_image = torch.randn(image_size, device=device) #start from pure noise
        for time in time_steps:
            pred_noise, current_image = self.forward(current_image, time)
        
        return current_image