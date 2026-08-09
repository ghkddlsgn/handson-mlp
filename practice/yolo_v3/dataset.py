# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("aladdinpersson/pascal-voc-dataset-used-in-yolov3-video", output_dir="./datasets/yolo",)

# print("Path to dataset files:", path)

from typing import Any

import numpy as np
import os
import pandas as pd
import torch
from torch import Tensor

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import iou_width_hegith as iou, non_max_suppresion as nms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, 
                 anchors, image_size=416, S=[13,26,52], c=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S #grid size
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) #means append
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index) -> Any:
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist(), 4, axis=1) #[class, x,y,w,h] -> [x,y,w,h,class]
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        #[p_0, p_x, y, w, h, c] s means grid num, c = [objectness, x, y, width, height, class]            
        targets = [torch.zeros((self.num_anchors//3, S, S, 6)) for S in self.S]
        
        for box in bboxes:
            iou_anchors:Tensor = iou(torch.tensor(box[2:4]), self.anchors)
            anchors_indices = iou_anchors.argsort(descending=True, dim=0) #return index, not element
            x,y,width,height,class_label = box
            has_anchor = [False, False, False]
            
            for anchor_idx in anchors_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale #0,1,2
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale #0,1,2