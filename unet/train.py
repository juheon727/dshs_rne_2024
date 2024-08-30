import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import json
import os
from dataset import Dataset
from unet import UNet
from eval import evaluate

if __name__ == '__main__':
    config = json.load(open('/app/unet/config.json'))
    model = UNet().cuda()
    #model.half()
    print(model)
    print("Number of Parameters:", sum(p.numel() for p in model.parameters()))
    img_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        #transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalization
    ])

    train_dataset = Dataset(
        path=config['train_dir'],
        img_resolution=config['img_resolution'],
        img_transforms=img_transforms,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    scaler = torch.cuda.amp.GradScaler()

    iou_max = 0
    epoch_max = 0
    loss_f = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    for epoch in tqdm(range(1, config['epochs']+1)):
        model.train()
        epoch_loss = 0
        for img, label in train_dataloader:
            optimizer.zero_grad()
            img, label = img.cuda(), label.cuda()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_hat = model(img)
                loss = loss_f(y_hat, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss
            
            del y_hat, img, label
        
        test_loss, iou = evaluate(model)

        with open(config['log_dir'], 'a') as f:
            print(f"[{epoch}/{config['epochs']}] Train Loss: {epoch_loss / len(train_dataloader):.4f}, Test Loss: {test_loss}, mIoU: {iou}", file=f)
        if iou > iou_max:
            iou_max = iou
            epoch_max = epoch
            torch.save(model.state_dict(), os.path.join(config['save_dir'], f'unet_best.pt'))
    
    with open(config['log_dir'], 'a') as f:
        print(f"[{epoch_max}/{config['epochs']}] Maximum mIoU: {iou}", file=f)