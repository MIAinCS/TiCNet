from config import net_config as config

from .module import *
from .multi_scale import conv_2nV1, conv_3nV1
from .transformer import build_transformer
from .position_encoding import build_position_encoding

bn_momentum = train_config['bn_momentum']


class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride=1, ):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FeatureNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32),
            Atten_Conv_Block(32),
        )

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        self.forw4 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        self.back1 = nn.Sequential(
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, out_channels),
            Atten_Conv_Block(out_channels),
        )

        self.back2 = nn.Sequential(
            ResBlock3d(192, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            Atten_Conv_Block(64),
        )

        self.back3 = nn.Sequential(
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            Atten_Conv_Block(128),
        )

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        self.position_embedding = build_position_encoding(config)
        self.transformer = build_transformer(config)

        self.scale1 = conv_3nV1(32, 64, 64)
        self.scale2 = conv_3nV1(64, 64, 64)
        self.scale3 = conv_2nV1(64, 64)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))

        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.preBlock(x)  # 24, 1/2
        out_pool = out
        out1 = self.forw1(out_pool)  # 32
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 64
        out3_pool, _ = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 64

        pe = self.position_embedding(out4)
        out4_tr = self.transformer(out4, pe) # 64

        out2_scale = self.scale1(out1, out2, out3) # 64
        out3_scale = self.scale2(out2, out3, out4) # 64
        out4_scale = self.scale3(out3, out4) # 64

        comb3 = self.back3(torch.cat((out4_tr, out4_scale), 1))
        rev2 = self.path1(comb3)
        comb2 = self.back2(torch.cat((rev2, out3_scale), 1))  # 96+96
        rev1 = self.path2(comb2)
        comb1 = self.back1(torch.cat((rev1, out2_scale), 1))  # 64+64

        return [x, rev2, comb1], out2


def build_feature_net():
    return FeatureNet(in_channels=1, out_channels=128)
