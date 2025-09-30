import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Small building blocks
# ----------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class DSConvBlock(nn.Module):
    def __init__(self, ch, hidden=None, num_layers=2):
        super().__init__()
        if hidden is None:
            hidden = ch
        layers = []
        for _ in range(num_layers):
            layers += [
                DepthwiseSeparableConv(ch, hidden, k=3),
                SE(hidden, r=8),
                DepthwiseSeparableConv(hidden, ch, k=3),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Correlation (local cost volume)
# ----------------------------

def local_correlation(f1: torch.Tensor, f2: torch.Tensor, radius: int) -> torch.Tensor:
    """
    f1, f2: (B, C, H, W)
    Returns: (B, (2r+1)^2, H, W)
    """
    B, C, H, W = f1.shape
    k = 2 * radius + 1
    f2_pad = F.pad(f2, (radius, radius, radius, radius))
    patches = F.unfold(f2_pad, kernel_size=k, padding=0, stride=1)  # (B, C*k*k, H*W)
    patches = patches.view(B, C, k * k, H, W)                        # (B, C, KK, H, W)
    corr = (f1.unsqueeze(2) * patches).sum(dim=1)                    # (B, KK, H, W)
    return corr / math.sqrt(C)

# ----------------------------
# Ultra-tiny learned refiner (optional)
# ----------------------------

class NanoRefine(nn.Module):
    """
    ~1–2k params residual flow touch-up. Very cheap.
    """
    def __init__(self, mid=16, dil=2):
        super().__init__()
        self.pre = DepthwiseSeparableConv(2, mid, k=3)
        self.dw  = nn.Conv2d(mid, mid, 3, padding=dil, dilation=dil, groups=mid, bias=False)
        self.pw  = nn.Conv2d(mid, mid, 1, bias=False)
        self.bn  = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.out = nn.Conv2d(mid, 2, 1)

    def forward(self, f):
        x = self.pre(f)
        x = self.act(self.bn(self.pw(self.dw(x))))
        return f + self.out(x)

# ----------------------------
# Flow Head (clean, tiny upsampling path)
# ----------------------------

class LiteFlowHead(nn.Module):
    """
    Lightweight optical-flow head for two feature maps.
    Inputs: feat1, feat2 -> (B, C, H, W)
    Output: flow at (B, 2, out_h, out_w)
    """
    def __init__(
        self,
        out_size: Tuple[int, int] = (640, 640),
        in_channels: int = 384,
        proj_channels: int = 128,
        radius: int = 4,
        fusion_channels: int = 256,
        fusion_layers: int = 2,
        refinement_layers: int = 1,    # feature-scale refinement
        use_nano_refine: bool = True, # ultra-tiny learned touch-up at full-res
    ):
        super().__init__()
        self.out_size = out_size
        self.radius = radius
        self.use_nano_refine = use_nano_refine

        # Project backbone features
        self.proj1 = nn.Conv2d(in_channels, proj_channels, 1, bias=False)
        self.proj2 = nn.Conv2d(in_channels, proj_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(proj_channels)
        self.bn2 = nn.BatchNorm2d(proj_channels)

        cv_ch = (2 * radius + 1) ** 2
        fuse_in = proj_channels * 3 + cv_ch
        self.fuse_in = nn.Sequential(
            DepthwiseSeparableConv(fuse_in, fusion_channels, k=3),
            SE(fusion_channels, r=8),
        )
        self.fuse_trunk = DSConvBlock(fusion_channels, hidden=fusion_channels, num_layers=fusion_layers)

        self.flow_head = nn.Sequential(
            DepthwiseSeparableConv(fusion_channels, fusion_channels, k=3),
            nn.Conv2d(fusion_channels, 2, 1)
        )

        if refinement_layers > 0:
            ref_in = fusion_channels + 2
            self.refine = nn.Sequential(
                DepthwiseSeparableConv(ref_in, fusion_channels, k=3),
                DSConvBlock(fusion_channels, num_layers=refinement_layers),
                nn.Conv2d(fusion_channels, 2, 1)
            )
        else:
            self.refine = None

        self.nano = NanoRefine(mid=12) if use_nano_refine else None  # even smaller mid

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        assert feat1.shape == feat2.shape, "feat1 and feat2 must have same shape"
        B, C, H, W = feat1.shape
        out_h, out_w = self.out_size

        # Projections
        f1 = F.silu(self.bn1(self.proj1(feat1)))
        f2 = F.silu(self.bn2(self.proj2(feat2)))

        # Correlation + fusion
        corr = local_correlation(f1, f2, self.radius)
        diff = torch.abs(f1 - f2)
        x = torch.cat([f1, f2, diff, corr], dim=1)
        x = self.fuse_in(x)
        x = self.fuse_trunk(x)

        # Coarse flow
        flow = self.flow_head(x)

        # Optional refinement at feature scale
        if self.refine is not None:
            flow = flow + self.refine(torch.cat([x, flow], dim=1))

        # Bilinear upsample with correct scaling (align_corners=False)
        flow = F.interpolate(flow, size=(out_h, out_w), mode='bilinear', align_corners=False)
        flow[:, 0] *= (out_w / W)
        flow[:, 1] *= (out_h / H)

        # Optional ultra-tiny learned touch-up
        if self.nano is not None:
            flow = self.nano(flow)

        return flow
    
if __name__ == "__main__":
    B, C, H, W = 4, 384, 40, 40
    img_size = (640, 640)
    f1 = torch.randn(B, C, H, W).to("cuda")  # DINOv3 feat at t
    f2 = torch.randn(B, C, H, W).to("cuda")  # DINOv3 feat at t+1

    of_head = LiteFlowHead(out_size = img_size, 
                           in_channels = 384,
                            proj_channels = 256,
                            radius = 4,                 # local search radius (2r+1)^2 cost channels
                            fusion_channels = 448,      # width of fusion trunk
                            fusion_layers = 3,          # depth in DSConv blocks
                            refinement_layers = 2).to("cuda")      # extra refinement after initial flow)

    # ----------------- Utility: parameter counting -----------------
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print parameter counts
    print('Depth params: ', count_parameters(of_head))


    flow_img = of_head(f1, f2)  # if your image is 320×320
    print(flow_img.shape)