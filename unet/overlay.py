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

def overlay_mask(image, mask):
    '''
    Overlay the mask on the image.
    image: PIL Image
    mask: numpy array of the mask with shape (H, W, 3)
    '''
    # Convert image to RGBA if not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Create an RGBA mask image with the same size as the input image
    mask_image = Image.fromarray(mask, mode='RGB')
    mask_image = mask_image.convert('RGBA')

    # Add alpha channel to the mask_image
    mask_alpha = Image.new('L', mask_image.size, 128)  # 50% opacity
    mask_image.putalpha(mask_alpha)
    
    # Combine the original image with the mask
    image_with_mask = Image.alpha_composite(image, mask_image)
    return image_with_mask


if __name__ == '__main__':
    config = json.load(open('/app/unet/config.json'))
    model = UNet().cuda()
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'unet_best.pt')))
    model.eval()
    print(model)
    print("Number of Parameters:", sum(p.numel() for p in model.parameters()))

    img_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        #transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalization
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
    model.eval()
    loss = 0
    iou = 0

    r, g, b = config['mask_color']

    for i, (img, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        img, label = img.cuda(), label.cuda()
        original_img = test_dataset.__getitem__(i)[0]  # Load original image
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_hat = model(img)
            loss = loss_f(y_hat, label)
            loss += loss / len(test_dataloader)
            iou += intersection_over_union(y_hat, label, threshold=config['positive_threshold']).item() / len(test_dataloader)
        
        y_hat_np = torch.sigmoid(y_hat).detach().cpu().numpy().squeeze()
        #y_hat_np = np.uint8(y_hat_np*255)
        
        # Convert to 3 channel image for overlay
        y_hat_rgb = np.stack([y_hat_np*r, y_hat_np*g, y_hat_np*b], axis=-1)
        y_hat_rgb = y_hat_rgb.astype(dtype=np.uint8)
        
        # Convert tensors back to PIL Image for overlaying
        original_pil_image = transforms.ToPILImage()(original_img.cpu())

        mask_overlay = overlay_mask(original_pil_image, y_hat_rgb)
        
        # Save the overlaid image
        mask_overlay.save(os.path.join(config['prediction_dir'], f'{i}.png'))

        del y_hat, img, label

    print('Loss: {:.4f}'.format(loss))
    print('mIoU: {:.4f}'.format(iou))
