# model/MDMFNet.py
import torch
import torch.nn as nn

from .CSPDRBNELAN import (
    Conv,
    RepConvN,
    DilatedReparamBlock,
    RepNBottleneck,
    DRBNBottleneck,
    RepNCSP,
    DRBNCSP,
    RepNCSPELAN4,
    CSPDRBNELAN,
)
from .HLFI import HLFI
from .MBRD import MBRD


class _Placeholder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


try:
    from ultralytics.nn.modules import ADown, AIFI, RepC3, Concat
except Exception:
    ADown = AIFI = RepC3 = Concat = _Placeholder

try:
    from ultralytics.nn.modules.head import RTDETRDecoder
except Exception:
    RTDETRDecoder = _Placeholder


def make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def width_scale(ch, w, max_ch):
    return make_divisible(min(ch, max_ch) * w, 8)


class MDMFBackbone(nn.Module):
    def __init__(self, w=1.0, max_ch=1024):
        super().__init__()
        c1 = width_scale(64, w, max_ch)
        c2 = width_scale(128, w, max_ch)
        c3 = width_scale(256, w, max_ch)
        c4 = width_scale(512, w, max_ch)
        c5 = width_scale(1024, w, max_ch)

        self.stem0 = Conv(3, c1, 3, 2)
        self.stem1 = Conv(c1, c2, 3, 2)

        self.stage2 = CSPDRBNELAN(c2, c2, c2 // 2, c2 // 4, 1, 13)

        self.down3 = MBRD(c2, c3)
        self.stage3 = CSPDRBNELAN(c3, c3, c3 // 2, c3 // 4, 1, 11)

        self.down4 = MBRD(c3, c4)
        self.stage4 = CSPDRBNELAN(c4, c4, c4 // 2, c4 // 4, 1, 9)

        self.down5 = MBRD(c4, c5)
        self.stage5 = CSPDRBNELAN(c5, c5, c5 // 2, c5 // 4, 1, 7)

        self.out_channels = [c3, c4, c5]

    def forward(self, x):
        x = self.stem0(x)
        x = self.stem1(x)
        x = self.stage2(x)
        p3 = self.stage3(self.down3(x))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))
        return [p3, p4, p5]


class MDMFHead(nn.Module):
    def __init__(self, nc=80, ch=(256, 512, 1024), hidden=256):
        super().__init__()
        c3, c4, c5 = ch

        self.input_proj2 = Conv(c5, hidden, 1, 1, act=False)
        self.aifi = AIFI(hidden, 8)
        self.lateral0 = Conv(hidden, hidden, 1, 1)

        self.input_proj1 = Conv(c4, hidden, 1, 1, act=False)
        self.fpn_block0 = RepC3(hidden * 2, hidden, 3)
        self.lateral1 = Conv(hidden, hidden, 1, 1)

        self.input_proj0 = Conv(c3, hidden, 1, 1, act=False)
        self.fpn_block1 = RepC3(hidden * 2, hidden, 3)

        self.down0 = MBRD(hidden, hidden)
        self.pan_block0 = RepC3(hidden * 2, hidden, 3)

        self.down1 = MBRD(hidden, hidden)
        self.pan_block1 = RepC3(hidden * 2, hidden, 3)

        self.proj_p3 = Conv(c3, hidden, 1, 1)
        self.proj_p4 = Conv(c4, hidden, 1, 1)
        self.proj_p5 = Conv(c5, hidden, 1, 1)

        self.sdfm0 = HLFI(hidden)
        self.sdfm1 = HLFI(hidden)
        self.sdfm2 = HLFI(hidden)

        self.decoder = RTDETRDecoder(
            nc=nc, ch=hidden, hidden_dim=256,
            num_queries=300, num_levels=3,
            num_decoder_layers=4, num_decoder_points=8, num_heads=8,
        )

    def forward(self, feats):
        p3, p4, p5 = feats

        x5 = self.input_proj2(p5)
        x5 = self.aifi(x5)
        y5 = self.lateral0(x5)

        u4 = nn.functional.interpolate(y5, scale_factor=2, mode='nearest')
        x4 = self.input_proj1(p4)
        f4 = self.fpn_block0(torch.cat([u4, x4], dim=1))
        y4 = self.lateral1(f4)

        u3 = nn.functional.interpolate(y4, scale_factor=2, mode='nearest')
        x3 = self.input_proj0(p3)
        f3 = self.fpn_block1(torch.cat([u3, x3], dim=1))

        d4 = self.down0(f3)
        pan4 = self.pan_block0(torch.cat([d4, y4], dim=1))
        d5 = self.down1(pan4)
        pan5 = self.pan_block1(torch.cat([d5, y5], dim=1))

        fp3 = self.proj_p3(p3)
        fp4 = self.proj_p4(p4)
        fp5 = self.proj_p5(p5)

        s0 = self.sdfm0([fp4, d4])
        s1 = self.sdfm1([fp5, d5])
        s2 = self.sdfm2([s0, fp3])

        out = self.decoder([s2, s1, s0])
        return out


class MDMFNet(nn.Module):
    def __init__(self, nc=80, w=1.0, max_ch=1024):
        super().__init__()
        self.backbone = MDMFBackbone(w=w, max_ch=max_ch)
        self.head = MDMFHead(nc=nc, ch=tuple(self.backbone.out_channels))

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


if __name__ == '__main__':
    model = MDMFNet(nc=80, w=1.0, max_ch=1024)
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    print('input :', x.shape)
    print('output:', out if isinstance(out, (tuple, list)) else out.shape)