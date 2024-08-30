import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from patches import classify_patches
import matplotlib.image

roads = classify_patches('road_detection.jpeg')
matplotlib.image.imsave('roadmap_prior.png', roads)
yp, xp = roads.shape

model = torch.hub.load('./yolov5/', 'custom', path='./yolov5/runs/train/exp5/weights/best.pt', source='local')

img = Image.open('road.jpeg').convert('RGB')
draw = ImageDraw.Draw(img)
w, h = img.size

results = model(img)
df = results.pandas().xyxy[0]
for _, row in df.iterrows():
    x1, y1, x2, y2, conf, cls, label = row
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
    for r in range(int(y1*yp/h), int(y2*yp/h)+1):
        for c in range(int(x1*xp/w), int(x2*xp/w)+1):
            roads[r][c] = 0

matplotlib.image.imsave('roadmap_after.png', roads)
img.save('road_detection.jpeg', 'JPEG')