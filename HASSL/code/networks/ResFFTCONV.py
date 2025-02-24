import  torch
from torch import  nn
import torch.nn.functional as F

class FFT(nn.Module):
    def __init__(self,channels,norm='backward'):
        super(FFT, self).__init__()
        # self.conv1x1_block = nn.Sequential(nn.Conv3d(2*channels,2*channels,1),nn.InstanceNorm3d(2*channels),nn.ReLU(inplace=True))
        self.conv1x1_block = nn.Conv3d(2*channels,2*channels,1)
        self.conv3x3_block = nn.Sequential(nn.Conv3d(channels, channels,3, 1,1), nn.BatchNorm3d(channels),nn.ReLU(inplace=True))

        self.norm = norm

    def forward(self,x):
        _,_,D,H,W = x.shape

        identity = x

        x_res = self.conv3x3_block(x)

        x_FFT = torch.fft.fftn(x,dim=(2,3,4),norm = self.norm)

        x_real = x_FFT.real
        x_img = x_FFT.imag

        x_cat = torch.cat([x_real,x_img],dim=1)
        x_cat = F.relu(self.conv1x1_block(x_cat))
        x_cat = self.conv1x1_block(x_cat)

        # for i in range(2):
        #     x_cat = self.conv1x1_block(x_cat)

        y_real,y_img = torch.chunk(x_cat,2,dim=1)

        y_FFT = torch.complex(real=y_real,imag=y_img)
        y_FFT = torch.fft.irfftn(y_FFT,dim=(2,3,4),s=(D,H,W),norm=self.norm)

        y = y_FFT + identity +x_res

        return y






