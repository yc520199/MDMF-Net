import torch.nn.functional as F
import torch
import torch.nn as nn



class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


class MBRD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.pre_dw = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)


        self.cut_c = Cut(in_channels, out_channels)


        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=2, dilation=2,
                      groups=min(in_channels, out_channels)),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )


        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.pool_conv = nn.Conv2d(in_channels * 2, out_channels, 1)
        self.pool_bn = nn.BatchNorm2d(out_channels)


        self.ca_mlp = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, 3, 1)
        )
        self.softmax = nn.Softmax(dim=1)

        self.fusion_final = nn.Conv2d(out_channels, out_channels, 1)
        self.act_final = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        x_pre = self.pre_dw(x)


        feat_c = self.cut_c(x)
        feat_x = self.conv_x(x_pre)


        feat_p = self.pool_bn(self.pool_conv(torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)))


        combined = torch.cat([feat_c, feat_x, feat_p], dim=1)

        weights = self.softmax(self.ca_mlp(nn.functional.adaptive_avg_pool2d(combined, 1)))

        weighted_feat = feat_c * weights[:, 0:1] + feat_x * weights[:, 1:2] + feat_p * weights[:, 2:3]

        return self.act_final(self.fusion_final(weighted_feat) + identity)