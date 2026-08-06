import torch
import torch.nn as nn
from  torch import Tensor

from study.d_01object_detect.bounding_box import multibox_prior

class down_sample_blk(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x:Tensor):
        result:Tensor = self.block2(self.block1(x))
        result = self.max_pool(result)
        return result

class SSD_SubBlock(nn.Module):
    def __init__(self, channel_dims:list[int], sizes:list[float], ratios:list[float], num_classes:int = 1,):
        super().__init__()
        if len(channel_dims) < 2: raise
        
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = len(sizes) + len(ratios) - 1
        
        self.layers = nn.Sequential(
            *[down_sample_blk(channel_dims[i-1], channel_dims[i]) for i in range(1, len(channel_dims))]
            )
        self.clf = nn.Conv2d(channel_dims[-1], self.num_anchors * (num_classes + 1), kernel_size=3, padding=1)
        self.bbox_predictor = nn.Conv2d(channel_dims[-1], self.num_anchors * 4, kernel_size=3, padding=1)
        
    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.layers(x)
        anchors = multibox_prior(h, self.sizes, self.ratios)
        clf_logit = self.clf(h)
        bbox_logit = self.bbox_predictor(h)
        
        return (h, anchors, clf_logit, bbox_logit)

class SSD(nn.Module):
    def __init__(self, num_classes:int = 1, Basenet_filters:list[int] = [3, 16, 32, 64],
                downsample_filters:list[int] = [64, 128, 128, 128, 128], 
                sizes:list[list[float]] = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],
                ratios:list[list[float]] = [[1, 2, 0.5]] * 5,
                ):
        super().__init__()
        
        num_anchors = [len(size) + len(ratio) - 1 for size, ratio in zip(sizes, ratios)]
        self.layers = [
            SSD_SubBlock(Basenet_filters, sizes[0], ratios[0], num_classes), #basenet
        ]
        for i in range(1, len(num_anchors)):
            self.layers.append(
                SSD_SubBlock([downsample_filters[i-1], downsample_filters[i-1]], sizes[i], ratios[i], num_classes)
            )
        
    def forward(self, x:Tensor):
        z:Tensor = x
        anchors, cls_preds, bbox_preds = [], [], []
        for layer in self.layers:
            z, anchor, clf, bbox = layer(z)
            anchors.append(anchor)
            cls_preds.append(clf)
            bbox_preds.append(bbox)
        
        return anchors, cls_preds, bbox_preds