import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models


class DownAndUp(nn.Module):
    def __init__(self,in_channels, out_channels):
       super(DownAndUp, self).__init__()
       temp = out_channels
       self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, temp, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
        )
    def forward(self, x):
     
        return self.conv1(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        

        x = torch.cat([x2, x1], dim=1)
        return x


# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels=3,n_classes=1):
       super(Model, self).__init__()
       self.img_channels = img_channels
       self.n_classes = n_classes
       # max_output = (input + 2 * padding - dilation(1 for default) * (kernel - 1) - 1) / stride + 1 
       self.maxpool = nn.MaxPool2d(kernel_size=2)
       self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
       self.down1 = DownAndUp(img_channels,64)
       self.down2 = DownAndUp(64,128)
       self.down3 = DownAndUp(128,256)
       self.down4 = DownAndUp(256,512)
       self.down5 = DownAndUp(512,512)
       
    def forward(self, x):
        x1 = self.down1(x)
     
        x2 = self.maxpool(x1)
        x3 = self.down2(x2)
   
        x4 = self.maxpool(x3)
        x5 = self.down3(x4)

        x6 = self.maxpool(x5)
        x7 = self.down4(x6)

        x8 = self.maxpool(x7)
        x9 = self.down5(x8)
    
        return x9
