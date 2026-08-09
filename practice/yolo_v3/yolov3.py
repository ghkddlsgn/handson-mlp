from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from yolov3_config import config

class CNNBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, use_batch_norm:bool = True, kernel_size:int = 3, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, kernel_size=kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm
    
    def forward(self, x:Tensor):
        if self.use_batch_norm:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)
        
class ResBlock(nn.Module):
    def __init__(self, channels:int, use_residual:bool = True, num_repeats:int = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                CNNBlock(channels, channels//2, kernel_size = 1),
                CNNBlock(channels//2, channels, kernel_size = 3, padding=1)
                )
            ]
            
        self.use_residual = use_residual
        self.num_repeats = num_repeats
    
    def forward(self, x:Tensor):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        
        return x
    
class ScalePrediction(nn.Module):
    def __init__(self, in_channels:int, num_classes:int):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, 3 * (num_classes + 5), False, kernel_size = 1) #3 bounding box, 5 -> [po, x,y,w,h]
        )
        self.num_classes = num_classes
    
    def forward(self, x:Tensor):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0,1,3,4,2)
        )
        # N * 3(anchors) * (13 * 13) * (5 + num_classes)

class YOLOv3(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=20) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
    
    def forward(self, x:Tensor):
        outputs = []
        route_connections = []
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x)
            
            if isinstance(layer, ResBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        
        return outputs
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(
                    in_channels, 
                    out_channels,
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=1 if kernel_size == 3 else 0
                    ))
                in_channels = out_channels        
            
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResBlock(in_channels, num_repeats=num_repeats))
            
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResBlock(in_channels, use_residual=False, num_repeats = 1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                    
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3 #why 3? previous layer res + current output + next layer
            
        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416  # YOLOv1: 448, YOLOv3: 416 (multi-scale training)

    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)

    assert out[0].shape == (
        2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5
    )
    assert out[1].shape == (
        2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5
    )
    assert out[2].shape == (
        2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5
    )

    print("Success!")