import torch
import torch.nn as nn

class DualConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(2):
            self.layers.append(nn.Conv2d(
                in_channels=in_channels if i==0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            ))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DualConvolutionBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convtranspose = nn.ConvTranspose2d(
            in_channels=2*in_channels, 
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv = DualConvolutionBlock(in_channels=in_channels*2, out_channels=out_channels)

    def forward(self, lowres, skip):
        x = self.convtranspose(lowres)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

if __name__ == '__main__':
    x = torch.randn(1, 1, 64, 64)

    encoder = EncoderBlock(in_channels=1, out_channels=64)
    enc_out, pool_out = encoder(x)

    conv = DualConvolutionBlock(64, 128)
    pool_out = conv(pool_out)
    print("Encoder output shape:", enc_out.shape)
    print("Max pooling output shape:", pool_out.shape)

    decoder = DecoderBlock(in_channels=64, out_channels=64)
    dec_out = decoder(pool_out, enc_out)
    print("Decoder output shape:", dec_out.shape)