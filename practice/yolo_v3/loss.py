import torch
import torch.nn as nn
from torch import Tensor
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss() #if there's more than 2 class, i need to use bce for multilable loss
        self.sigmoid = nn.Sigmoid()
        
        #constants
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
    
    def forward(self, predictions:Tensor, target:Tensor, anchors:Tensor):
        """
        target:
            dimension = (batch, anchor, grid_y, grid_x, 6)
            6 -> [objectness, x, y, width, height, class]
        """
        
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        
        #No object loss
        no_object_loss = self.bce(
            predictions[..., 0:1][noobj], target[..., 0:1][noobj]
        )
        #object loss
        """
        original anchors shape 
        [
            [anchor1_w, anchor1_h],
            [anchor2_w, anchor2_h],
            [anchor3_w, anchor3_h]
        ]
        
        box_predictions[..., 2:].shape == (N, 3, S, S, 2) == (N, anchor, grid_y, grid_x, width/height)
        """
        anchors = anchors.reshape(1,3,1,1,2) # 3 * 2 (anchor scales, h,w), p_w * exp(t_w)
        box_preds = torch.cat(
            [self.sigmoid(predictions[..., 1:3]), #for limit between 0~1
             torch.exp(predictions[..., 3:5]) * anchors #with exp, always positive. 3:5->(w,h)
            ], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce(
            predictions[..., 0:1][obj],
            ious * target[..., 0:1][obj],
        )
        
        #box coordinate loss
        box_predictions = torch.cat(
            [self.sigmoid(predictions[..., 1:3]), predictions[..., 3:5]],
            dim=-1,
        )
        box_targets = torch.cat(
            [
                target[..., 1:3],
                torch.log(1e-16 + target[..., 3:5] / anchors),
            ],
            dim=-1,
        )
        box_loss = self.mse(box_predictions[obj], box_targets[obj])
        
        #class loss
        class_loss = self.entropy(
            predictions[..., 5:][obj], target[..., 5][obj].long()
        )
        
        return (
            self.lambda_box * box_loss 
            + self.lambda_obj * object_loss 
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
