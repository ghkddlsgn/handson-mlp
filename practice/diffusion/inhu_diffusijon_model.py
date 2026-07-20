import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
from torch import Tensor
from practice.diffusion.inhu_res_unet import InhuResUnet

class InhuDiffusionModel(nn.Module):
    def __init__(self, unet:InhuResUnet, input_dim:int, output_dim:int, time_dim:int=256, dropout:float=0.1, num_groups:int=8, total_timestep:int=4000):
        super().__init__()
        self.s:float = 0.008
        self.b_max:float = 0.999
        self.total_timestep = total_timestep
        
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        self.unet:InhuResUnet = unet
        
    #noise_image = [b, rgb, w, h], time_steps[b, image, time_steps]
    def forward(self, noise_image:Tensor, time_steps:Tensor):
        embeded_time = self.time_mlp(self.sinusoidal_embedding(time_steps)) #[B, time_dim]
        pred_noise = self.unet(noise_image, embeded_time)
        a_t_bar = self.get_a_t_bar(time_steps)
        pred_original_image:Tensor = self.get_original_image(pred_noise, noise_image, a_t_bar)
    
    def sinusoidal_embedding(self, time_steps: Tensor) -> Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(-torch.arange(half, device=time_steps.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = time_steps[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    #predicted_noise = [b, rgb, w, h], noised_image = same, a_t_bar = [b,image,time_steps]
    def get_original_image(self, predicted_noise:Tensor, noised_image:Tensor, a_t_bar:Tensor) -> Tensor:
        original_image:Tensor = (noised_image - torch.sqrt(1 - a_t_bar) * predicted_noise) / torch.sqrt(a_t_bar)
        return original_image
        
    #generate noise image --------------------------------------
    def get_noised_image(self, target_time_steps:Tensor, original_image:Tensor, 
                         noise:Tensor, a_t_bar:Tensor) -> Tensor:
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
