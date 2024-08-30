import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import json
import os
from PIL import Image
import numpy as np
from dataset import Dataset
from unet import UNet

def intersection_over_union(y_hat: torch.tensor, labels: torch.tensor, threshold=0.5):
    '''
    Calculates the Intersection over Union between the prediction and labels
    y_hat: logit output of UNet
    labels: ground truth labels
    Returns:
        mIoU: Mean Intersection over Union score
    '''

    y_hat = torch.sigmoid(y_hat)
    y_hat_bin = (y_hat > threshold).float()
    
    intersection = torch.sum(y_hat_bin * labels, dim=(0, 2, 3))
    union = torch.sum(y_hat_bin + labels, dim=(0, 2, 3)) - intersection
    
    # Avoid division by zero
    union = torch.clamp(union, min=1e-6)
    
    # Compute IoU
    iou = intersection / union
    
    return iou

def evaluate(model):
    config = json.load(open('/app/unet/config.json'))
    model.eval()
    img_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalization
    ])

    test_dataset = Dataset(
        path=config['test_dir'],
        img_resolution=config['img_resolution'],
        img_transforms=img_transforms,
        zero_index=81
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers']
    )

    loss_f = nn.BCEWithLogitsLoss(reduction='sum')
    loss = 0
    iou = 0
    for i, (img, label) in enumerate(test_dataloader):
        img, label = img.cuda(), label.cuda()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_hat = model(img)
            loss = loss_f(y_hat, label)
            loss += loss/len(test_dataloader)
            iou += intersection_over_union(y_hat, label, threshold=config['positive_threshold']).item()/len(test_dataloader)
        y_hat_np = torch.sigmoid(y_hat).detach().cpu().numpy().squeeze()
        y_hat_np = np.uint8(y_hat_np*255)
        
        del y_hat, img, label

    return loss, iou
        
