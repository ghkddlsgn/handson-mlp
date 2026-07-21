# import torch
# import torch.nn as nn
# from torch import Tensor
# from typing import Tuple

# s:float = 0.008
# b_max:float = 0.999
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_noised_image(target_time_steps:Tensor, original_image:Tensor, 
#                      total_timestep:int) -> Tuple[Tensor, Tensor]:
#     target_time_steps = target_time_steps.to(device)
#     original_image = original_image.to(device)
#     noise = torch.randn_like(original_image, device=device)
#     a:Tensor = get_a_t_bar(target_time_steps, total_timestep)[:, None,None,None]
#     noised_image:Tensor = original_image * torch.sqrt(a) + (torch.sqrt(1 - a) * noise)
#     return noised_image, noise

# def cos_func(time_steps:Tensor, total_timestep:int) -> Tensor:
#     time_steps = time_steps.to(device)
#     nominator = (time_steps / total_timestep + s) * torch.pi
#     denominator = (1 + s) * 2
#     return torch.cos(nominator / denominator) ** 2

# def get_a_t_bar(time_steps:Tensor, total_timestep:int) -> Tensor:
#     time_steps = time_steps.to(device)
#     return cos_func(time_steps, total_timestep) / cos_func(torch.zeros_like(time_steps), total_timestep)

# def get_b_t(time_steps:Tensor, a_t_bar:Tensor, a_t_bar_past:Tensor) -> Tensor:
#     a_t_bar = a_t_bar.to(device)
#     a_t_bar_past = a_t_bar_past.to(device)
#     return 1 - (a_t_bar / a_t_bar_past)