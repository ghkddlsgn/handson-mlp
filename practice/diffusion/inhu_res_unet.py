import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



class InhuResBlock(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, time_dim:int=256, dropout:float=0.1, num_groups:int=8, num_heads:int = 0):
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
        
        self.residual_proj = nn.Identity() if input_dim == output_dim else nn.Conv2d(input_dim, output_dim, 1)
        
        #if num heads < 1, no self attention
        self.use_attention:bool = (num_heads >= 1)
        if self.use_attention:
            self.norm3 = nn.GroupNorm(num_groups, output_dim)
            self.attention = nn.MultiheadAttention(output_dim, num_heads, dropout, batch_first=True)
        
    def forward(self, input:Tensor, embeded_time:Tensor):        
        h1 = self.norm1(input)
        h1 = F.gelu(h1)
        h1 = self.conv1(h1) #[batch, output_dim, h, w]
        
        scale, shift = self.time_proj(embeded_time).chunk(2, dim=1) # each of them is [batch, outputdim]
        
        scale, shift = scale[:, :, None, None], shift[:, :, None, None]
        
        h2 = self.norm2(h1)
        h2 = h2 * (1 + scale) + shift
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)
        h2 = self.conv2(h2)
        
        h3 = h2 + self.residual_proj(input)

        if not self.use_attention:
            return h3
        else:
            return self.self_attention(h3)
    
    def self_attention(self, x:Tensor) -> Tensor:
        b,c,h,w = x.shape
        x_flat:Tensor = self.norm3(x).flatten(2) #[B, c, h*w]
        x_flat = x_flat.transpose(1,2) #[B, h*w, c]
        
        attention_output, _ = self.attention(x_flat, x_flat, x_flat) #[B, h*w, c]
        attention_output = attention_output.transpose(1,2).unflatten(2, (h,w)) #[B,c,h,w]
        return x + attention_output

class InhuDownsample(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.conv(x)

class InhuUpsample(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x:Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

class InhuResUnet(nn.Module):
    def __init__(self, time_dim:int = 256, image_dim:int = 3, channels:list[int] = [64, 128, 256, 512, 512, 512, 1024], 
                 attention_start_idx:int = 4, num_heads:int = 8, dropout:float = 0.1, num_groups:int = 8):
        super().__init__()
        
        no_attention = not ( 0 <= attention_start_idx < len(channels) - 1)

        if no_attention:
            attention_start_idx = len(channels) - 1
            num_heads = 0
        
        self.init_conv = nn.Conv2d(image_dim, channels[0], 3, 1, 1)
        self.encoder_layers:nn.ModuleList = nn.ModuleList([])
        self.downsamples:nn.ModuleList = nn.ModuleList([])
        
        mid_channels = channels[-1]
        self.mid_block1 = InhuResBlock(mid_channels, mid_channels, time_dim, dropout, num_groups, num_heads)
        self.mid_block2 = InhuResBlock(mid_channels, mid_channels, time_dim, dropout, num_groups, num_heads)
        
        self.decoder_layers:nn.ModuleList = nn.ModuleList([])
        self.upsamples:nn.ModuleList = nn.ModuleList([])
        
        self.output_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels[0])
        self.output_conv = nn.Conv2d(channels[0], image_dim, 3, 1, 1)
        
        
        for i in range(0, attention_start_idx):
            self.encoder_layers.append(InhuResBlock(channels[i], channels[i + 1], time_dim, dropout, num_groups))
            self.downsamples.append(InhuDownsample(channels[i + 1], channels[i + 1]))
        
        for i in range(attention_start_idx, len(channels) - 1):
            self.encoder_layers.append(InhuResBlock(channels[i], channels[i + 1], time_dim, dropout, num_groups, num_heads))
            self.downsamples.append(InhuDownsample(channels[i + 1], channels[i + 1]))
        
        for i in range(len(channels) - 2, attention_start_idx - 1, -1):
            output_dim = channels[i]
            self.upsamples.append(InhuUpsample(channels[i + 1], channels[i + 1]))
            self.decoder_layers.append(InhuResBlock(channels[i + 1] * 2, output_dim, time_dim, dropout, num_groups, num_heads))
        
        for i in range(attention_start_idx - 1, -1, -1):
            output_dim = channels[i]
            self.upsamples.append(InhuUpsample(channels[i + 1], channels[i + 1]))
            self.decoder_layers.append(InhuResBlock(channels[i + 1] * 2, output_dim, time_dim, dropout, num_groups))
        
        # 64 : 512
        # 128 : 256
        # 256 : 128
        # 512 : 64
        # 512 : 32
        # 512 : 16
        # 1024 : 8
    
    def forward(self, x:Tensor, embedded_times:Tensor):
        encoder_outputs:list[Tensor] = []
        input = self.init_conv(x)
        
        #encode part
        for enc_layer, downsample in zip(self.encoder_layers, self.downsamples):
            layer_output = enc_layer(input, embedded_times)
            encoder_outputs.append(layer_output)
            
            input = downsample(layer_output)
        
        input = self.mid_block1(input, embedded_times)
        input = self.mid_block2(input, embedded_times)
        
        #decode part
        for dec_layer, upsample, skip in zip(self.decoder_layers, self.upsamples, reversed(encoder_outputs)):
            input = upsample(input)
            
            if input.shape[-2:] != skip.shape[-2:]:
                input = F.interpolate(input, size=skip.shape[-2:]) #interpolate only accepts 2d
            
            concat_input = torch.concat([input, skip], dim=1)
            input = dec_layer(concat_input, embedded_times)
        
        h = self.output_norm(input)
        h = F.gelu(h)
        
        predicted_noise = self.output_conv(h)
        
        return predicted_noise
