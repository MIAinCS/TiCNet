
import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from config import train_config



class SE_Block(nn.Module):
    def __init__(self, channel):
        super(SE_Block, self).__init__()
        self.Global_Pool = nn.AdaptiveAvgPool3d(1)
        self.FC1 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.ReLU(), )
        self.FC2 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.Sigmoid(), )

    def forward(self, x):
        G = self.Global_Pool(x)
        G = G.view(G.size(0), -1)
        fc1 = self.FC1(G)
        fc2 = self.FC2(fc1)
        fc2 = torch.unsqueeze(fc2, 2)
        fc2 = torch.unsqueeze(fc2, 3)
        fc2 = torch.unsqueeze(fc2, 4)
        return fc2 * x


class Atten_Conv_Block(nn.Module):
    def __init__(self, channel):
        super(Atten_Conv_Block, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm3d(channel, momentum=train_config['bn_momentum']),
                                   nn.ReLU(inplace=True))
        self.adppool = nn.AdaptiveAvgPool3d(1)
        self.rw_conv = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm3d(channel))
        self.fc_adapt_channels = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=3, padding=1, bias=True),
                                               nn.Sigmoid())

    def forward(self, x):
        out = self.relu(x)
        out1 = self.conv1(out)
        out1 = self.relu(out1)
        out1_pool = self.adppool(out1)

        out2 = self.rw_conv(out1_pool)
        out3 = self.fc_adapt_channels(out2)

        return out3 * x


class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3, 4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3, 4), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool3d(x, 2, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, d, h, w = x.size()

        n = d * w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class SRM(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRM, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1, 1)
        return x * g.expand_as(x)


class ECALayer(nn.Module):
    """
    Constructs a ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class External_attention(nn.Module):

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv3d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        # self.linear_1.weight = self.linear_0.weight.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv3d(c, c, 1, bias=False),
            nn.BatchNorm3d(c))

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, d, h, w = x.size()
        n = d * h * w
        x = x.view(b, c, d * h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))  # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, d, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x


class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio=1 / 4):
        super(GlobalContextBlock, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1, 1]),
            nn.ReLU(),  # yapf: disable
            nn.Conv3d(self.planes, self.inplanes, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, depth, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, depth * height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, depth * height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term

        return out


class SelfAttention(nn.Module):
    """ self attention module"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query = nn.Conv3d(in_channels=in_dim,
                               out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv3d(in_channels=in_dim,
                             out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv3d(in_channels=in_dim,
                               out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, depth, height, width = x.size()
        proj_query = self.query(x).reshape(
            m_batchsize, -1, depth * width * height).permute(0, 2, 1)
        proj_key = self.key(x).reshape(m_batchsize, -1, depth * width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).reshape(m_batchsize, -1, depth * width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.reshape(m_batchsize, C, depth, height, width)

        return out


class DoubleAtten(nn.Module):

    def __init__(self, in_c):
        super(DoubleAtten, self).__init__()
        self.in_c = in_c
        self.convA = nn.Conv3d(in_c, in_c, kernel_size=1)
        self.convB = nn.Conv3d(in_c, in_c, kernel_size=1)
        self.convV = nn.Conv3d(in_c, in_c, kernel_size=1)

    def forward(self, input):
        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, d, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, d * h * w)
        atten_map = atten_map.view(b, self.in_c, 1, d * h * w)
        global_descriptors = torch.mean(
            (feature_maps * F.softmax(atten_map, dim=-1)), dim=-1)

        v = self.convV(input)
        atten_vectors = F.softmax(
            v.view(b, self.in_c, d * h * w), dim=-1)
        out = torch.bmm(atten_vectors.permute(0, 2, 1),
                        global_descriptors).permute(0, 2, 1)

        return out.view(b, _, d, h, w)


class CascadeMSC(nn.Module):
    def __init__(self, n_in, n_out):
        super(CascadeMSC, self).__init__()
        self.Conv5x5 = nn.Conv3d(
            n_in, n_out, kernel_size=5, stride=1, padding=2)
        self.Conv3x3 = nn.Conv3d(
            n_out, n_out, kernel_size=3, stride=1, padding=1)
        self.Conv1x1_1 = nn.Conv3d(n_out, n_out, kernel_size=1, stride=1)
        self.Conv1x1_2 = nn.Conv3d(n_out*3, n_out, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.Conv5x5(x)
        x2 = self.Conv3x3(sum(x, x1))
        x3 = self.Conv1x1_1(sum(x, x2))
        y = self.Conv1x1_2(torch.cat((x1, x2, x3), 1))

        return y


class DSPA(nn.Module):
    def __init__(self, n_in, n_out):
        super(DSPA, self).__init__()
        self.dilated_conv1 = nn.Conv1d(
            n_in, n_out, kernel_size=3, stride=1, padding=1, dilation=2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, d, h, w = x.size()
        x1 = x.view(b, c, -1)  # B x C x N
        x1_t = torch.transpose(x1, 1, 2)  # B x N x C
        x2 = self.dilated_conv1(x1)  # B x C x K
        x3 = self.softmax(torch.bmm(x1_t, x2))  # B x N x K
        x3_t = torch.transpose(x3, 1, 2)  # B x K x N
        x4 = torch.bmm(x2, x3_t)  # B x C x N
        out = x1 + x4
        out = out.view(b, c, d, h, w)
        return out


class CA(nn.Module):
    def __init__(self, n_in, n_out):
        super(CA, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, d, h, w = x.size()
        x1 = x.view(b, c, -1)  # B x C x N
        x1_t = torch.transpose(x1, 1, 2)  # B x N x C
        x2 = self.softmax(torch.bmm(x1, x1_t))  # B x C x C
        x2_t = torch.transpose(x2, 1, 2)  # B x C x C
        x3 = torch.bmm(x2_t, x1)  # B x C x N
        out = x1 + x3
        out = out.view(b, c, d, h, w)

        return out


class MDSA(nn.Module):
    def __init__(self, n_in, n_out):
        super(MDSA, self).__init__()
        self.dspa = CBAM(n_in, n_out)
        self.ca = SRM(n_out, n_out)

    def forward(self, x):
        x_dspa = self.dspa(x)
        x_ca = self.ca(x_dspa)

        return x_ca


class MDSAwithCascadeMSC(nn.Module):
    def __init__(self, n_in, n_out):
        super(MDSAwithCascadeMSC, self).__init__()
        self.dspa = CBAM(n_in, n_out)
        self.msc = CascadeMSC(n_out, n_out)
        self.ca = GCT(n_out)

    def forward(self, x):
        x_dspa = self.dspa(x)
        x_msc = self.msc(x_dspa)
        x_ca = self.ca(x_msc)

        return x_ca



if __name__ == '__main__':
    net = MDSAwithCascadeMSC(64, 64)

    input = torch.rand([4, 64, 32, 32, 32])
    output = net(input).cuda()
