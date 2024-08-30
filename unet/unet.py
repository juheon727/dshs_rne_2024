import torch
import torch.nn as nn
from unetmodules import *

class UNet(nn.Module):
    def __init__(self):
        '''
        Implementation of the UNet Semantic Segmentation Network
        '''
        super().__init__()
        # Encoder
        self.encoder1 = EncoderBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = DualConvolutionBlock(512, 1024)
        
        # Decoder
        self.decoder4 = DecoderBlock(512, 512)
        self.decoder3 = DecoderBlock(256, 256)
        self.decoder2 = DecoderBlock(128, 128)
        self.decoder1 = DecoderBlock(64, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        enc1, pool1 = self.encoder1(x)
        enc2, pool2 = self.encoder2(pool1)
        enc3, pool3 = self.encoder3(pool2)
        enc4, pool4 = self.encoder4(pool3)
        bottleneck = self.bottleneck(pool4)
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        
        out = self.final_conv(dec1)
        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 64, 64)  # batch_size=1, channels=1, height=64, width=64

    model = UNet()
    print(model)
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("Number of Parameters:", sum(p.numel() for p in model.parameters()))