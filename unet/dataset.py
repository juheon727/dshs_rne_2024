import torch
import torch.nn as nn
import torch.utils.data
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, img_resolution=448, img_transforms=None, zero_index=0):
        self.path = path
        assert len(os.listdir(os.path.join(self.path, 'images'))) == len(os.listdir(os.path.join(self.path, 'labels')))
        self.img_transforms = img_transforms
        self.img_resolution = img_resolution
        self.zero_index = zero_index

    def __len__(self):
        return len(os.listdir(os.path.join(self.path, 'images')))
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, 'images/um_{:06d}.png'.format(idx + self.zero_index))).convert('RGB')
        img = img.resize((self.img_resolution, self.img_resolution))
        mask = Image.open(os.path.join(self.path, 'labels/um_lane_{:06d}.png'.format(idx + self.zero_index))).convert('RGB')
        mask = np.array(mask.resize((self.img_resolution, self.img_resolution)), dtype=np.float16)
        mask = np.where(mask.sum(axis=-1) > 255, 1., 0.)
        mask = np.expand_dims(mask, axis=0)

        if self.img_transforms is not None:
            img = self.img_transforms(img)

        return img.half(), mask

if __name__ == '__main__':
    img_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        #transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])  # Normalization
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Dataset(
        path='/app/lanesegmentation_data/train',
        img_transforms=img_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4
    )
    
    print(len(train_dataset))

    # Optionally, test DataLoader
    for images, masks in train_loader:
        #images, masks = images.half(), masks.half()
        print(images[0])
        #print(masks[0])
        print(images.sum())  # Should be [batch_size, 3, 448, 448]
        print(masks.shape)   # Should be [batch_size, 1, 448, 448]
        #break
