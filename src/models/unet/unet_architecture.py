import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, inputs):
        return self.conv(inputs)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        )
        self.conv = DoubleConv(out_c*2, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class DownscalingUnet(nn.Module):
    def __init__(self, in_ch=7, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.e1 = EncoderBlock(in_ch, features[0])
        self.e2 = EncoderBlock(features[0], features[1])
        self.e3 = EncoderBlock(features[1], features[2])
        self.e4 = EncoderBlock(features[2], features[3])         
        self.b = DoubleConv(features[3], features[3]*2)         
        self.d1 = DecoderBlock(features[3]*2, features[3])
        self.d2 = DecoderBlock(features[3], features[2])
        self.d3 = DecoderBlock(features[2], features[1])
        self.d4 = DecoderBlock(features[1], features[0])         
        self.outputs = nn.Conv2d(features[0], out_ch, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        return self.outputs(d4)

    def last_layer(self):
        return self.outputs
