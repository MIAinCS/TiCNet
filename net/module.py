
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from config import train_config



class Atten_Conv_Block(nn.Module):
    def __init__(self, channel):

        super(Atten_Conv_Block, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channel, momentum=train_config['bn_momentum']),
            nn.ReLU(inplace=True)
        )
        self.adppool = nn.AdaptiveAvgPool3d(1)
        self.rw_conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm3d(channel)
        )
        self.fc_adapt_channels = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.relu(x)
        out1 = self.conv1(out)
        out1 = self.relu(out1)
        out1_pool = self.adppool(out1)

        out2 = self.rw_conv(out1_pool)
        out3 = self.fc_adapt_channels(out2)

        return out3 * x
