import torch
import torch.nn as nn
from torch import Tensor
from practice.diffusion.inhu_res_unet import InhuResUnet
from practice.diffusion.inhu_diffusion_config import DIFFUSION_MODEL_CONFIG
import math
import matplotlib.pyplot as plt
from typing import Any, TypedDict, Tuple, Optional, Sequence

NOISE_OFFSET: float = 0.008
EPS:float = 1e-8

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
    def __init__(
        self,
        unet: InhuResUnet,
        time_dim: int = DIFFUSION_MODEL_CONFIG["TIME_DIM"],
        total_timestep: int = DIFFUSION_MODEL_CONFIG["TOTAL_TIME"],
    ):
        super().__init__()

        self.total_timestep = total_timestep
        
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        self.b_max:float = 0.999
        
        self.unet:InhuResUnet = unet

    def get_device(self):
        return next(self.parameters()).device
        
    #noise image generator -----------------------------------------------------------------------------------
    def get_noised_image(self, target_time_steps:Tensor, original_image:Tensor, 
                         noise:Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        device = self.get_device()
        target_time_steps = target_time_steps.to(device=device)
        original_image = original_image.to(device=device)
        
        if noise is None:
            noise = torch.randn_like(original_image)
        noise = noise.to(device=device)
        
        a_t_bar = self.get_a_t_bar(target_time_steps)
        a_t_bar = a_t_bar.to(device=self.get_device())
        a_t_bar = a_t_bar[:, None, None, None]
        noised_image = original_image * torch.sqrt(a_t_bar) + torch.sqrt(1 - a_t_bar) * noise
        return noised_image, noise

    def cos_func(self, time_steps:Tensor) -> Tensor:
        nominator = (time_steps / self.total_timestep + NOISE_OFFSET) * torch.pi
        denominator = (1 + NOISE_OFFSET) * 2
        return torch.cos(nominator / denominator) ** 2

    def get_a_t_bar(self, time_steps:Tensor) -> Tensor:
        return self.cos_func(time_steps) / self.cos_func(torch.zeros_like(time_steps))

    def get_b_t(self, a_t_bar:Tensor, a_t_bar_past:Tensor) -> Tensor:
        return 1 - (a_t_bar / a_t_bar_past)  
    #noise image generator end ----------------------------------------------------------------------------------
    
    #noise_image = [b, rgb, w, h], time_steps[b]
    def forward(self, noised_image:Tensor, time_steps:Tensor):
        embeded_time = self.time_mlp(self.sinusoidal_embedding(time_steps)) #[B]
        pred_noise = self.unet(noised_image, embeded_time)
        a_t_bar = self.get_a_t_bar(time_steps)
        pred_original_image:Tensor = self.get_original_image(pred_noise, noised_image, a_t_bar)
        
        return pred_noise, pred_original_image
            
    def sinusoidal_embedding(self, time_steps: Tensor) -> Tensor: #[b, time_embed]
        half = self.time_dim // 2
        freqs = torch.exp(-torch.arange(half, device=self.get_device()) * math.log(10000.0) / (half - 1))
        args = time_steps[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    #predicted_noise = [b, rgb, w, h], noised_image = same, a_t_bar = [b]
    def get_original_image(self, predicted_noise:Tensor, noised_image:Tensor, a_t_bar:Tensor)-> Tensor:
        a_t_bar = a_t_bar.to(device=self.get_device(), dtype=noised_image.dtype)
        a_t_bar = a_t_bar[:, None, None, None]
        a_t_bar = a_t_bar.clamp(min=1e-8, max=1.0)
        original_image:Tensor = (noised_image - torch.sqrt(1 - a_t_bar) * predicted_noise) / torch.sqrt(a_t_bar)
        return original_image

    def generate_image(self, image_size:list[int] = [1, 3, 64, 64], steps:int = 40, seed:int=42) -> tuple[list[Tensor], list[Tensor]]:
        self.eval()
        torch.manual_seed(seed)
        time_steps = torch.linspace(self.total_timestep - 1, 0, steps, dtype=torch.long, device=self.get_device())
        
        with torch.no_grad():        
            pred_original_image_history:list[Tensor] = []
            step_history:list[Tensor] = []
            current_image:Tensor = torch.randn(image_size, device=self.get_device()) #start from pure noise
            batch_size = current_image.size(0)
            
            for start_time, target_time in zip(time_steps[:-1], time_steps[1:]):
                start_time = start_time.expand(batch_size)
                target_time = target_time.expand(batch_size)
                pred_noise, pred_original_image = self.forward(current_image, start_time)
                pred_original_image = pred_original_image.clamp(-1, 1)
                
                next_noise_image, next_noise = self.get_noised_image(target_time, pred_original_image, pred_noise)
                current_image = next_noise_image #for next step
                
                pred_original_image_history.append(pred_original_image)
                step_history.append(current_image)
            
        return pred_original_image_history, step_history

    def show_tensor_image(self, image_list:Sequence[Tensor] | tuple[list[Tensor], list[Tensor]], 
                          max_col:int = 5, step:Optional[list[int]] = None, titles:Optional[Sequence[str]] = None):
        if isinstance(image_list, tuple):
            image_list = image_list[0]

        image_count = len(image_list)
        col_count = min(max_col, image_count)
        row_count = math.ceil(image_count / col_count)

        fig, axes = plt.subplots(
            row_count,
            col_count,
            figsize=(col_count * 3, row_count * 3),
        )

        if image_count == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        if step is None:
            step = [idx for idx in range(len(image_list))]

        if titles is not None and len(titles) != image_count:
            raise ValueError("titles must contain one title for each image")

        for idx, image in enumerate(image_list):
            image = image.detach().cpu()

            # if batch image: [B, C, H, W] -> take first image
            if image.dim() == 4:
                image = image[0]

            # [C, H, W] -> [H, W, C]
            image = image.permute(1, 2, 0)

            # convert [-1, 1] -> [0, 1]
            image = (image + 1) / 2
            image = image.clamp(0, 1)

            axes[idx].imshow(image)
            axes[idx].set_title(titles[idx] if titles is not None else f"{step[idx]}")
            axes[idx].axis("off")

        # hide empty subplots
        for idx in range(image_count, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()
