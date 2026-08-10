from torch.utils.data import DataLoader

import config
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import YOLODataset
from yolov3 import YOLOv3
from tqdm import tqdm #for progress bar
from utils import (
    mean_average_precision,
    cells_to_bboxes, #convert coordinate from cell(relative) to entier image
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss

torch.backends.cudnn.benchmark = True

def train_fn(train_loader:DataLoader, model:nn.Module, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True) #progress bar
    losses = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )
        
        with torch.autocast('cuda'):
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
                
            )
        
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss = mean_loss)
        
        

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.GradScaler()
    
    
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET / "100examples.csv",
        test_csv_path=config.DATASET / "100examples.csv",
    )
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    
    scaled_anchors = (
        torch.tensor(config.ANCHORS) * torch.tensor(config.S).view(3,1,1)
    ).to(config.DEVICE)
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)

if __name__ == "__main__":
    main()