import torch.nn.functional as F
import torch
import torch.nn as nn

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-padding
    return p

class FSI_Recalibrate(nn.Module):

    def __init__(self, channels, r=4):
        super().__init__()
        inter = max(channels // r, 32)


        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)


        self.local_spat = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            Conv(2 * channels, 2 * channels, k=3, g=2 * channels)  # 深度可分离卷积
        )


        self.weight_gen = nn.Sequential(
            Conv(6 * channels, inter, k=1),
            nn.Conv2d(inter, 2 * channels, kernel_size=1),  # 最后一层通常不用 BN 和激活，直接输出权重
            nn.Sigmoid()
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)  # [B, 2*C, H, W]


        f_avg = self.global_avg(inp)
        f_max = self.global_max(inp)


        f_spat = self.local_spat(inp)
        f_spat = F.adaptive_avg_pool2d(f_spat, 1)


        combined = torch.cat([f_avg, f_max, f_spat], dim=1)


        gate = self.weight_gen(combined)
        return gate


class HLFI(nn.Module):

    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = max(channels // r, 32)

        self.Recalibrate = FSI_Recalibrate(channels, r)
        self.channel_agg = Conv(2 * channels, channels, k=1)


        self.local_att = nn.Sequential(
            Conv(channels, inter_channels, k=1),
            Conv(inter_channels, channels, k=1, act=False),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(channels, inter_channels, k=1),
            Conv(inter_channels, channels, k=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x1, x2 = data
        B, C, H, W = x1.shape


        recal_w = self.Recalibrate(x1, x2)
        inp = torch.cat([x1, x2], dim=1)


        recal_input = inp * recal_w + inp
        x1_r, x2_r = torch.split(recal_input, C, dim=1)


        agg = self.channel_agg(recal_input)


        gate = self.sigmoid(self.local_att(agg) * self.global_att(agg))


        out = gate * x1_r + (1 - gate) * x2_r
        return out
