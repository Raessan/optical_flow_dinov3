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
    
class ConvexUpsampler(nn.Module):
    """
    RAFT-style convex upsampler.
    Given a low-res flow (B,2,H,W) and features (B,C,H,W),
    predicts a mask (B, 9*up*up, H, W) and upsamples flow to (B,2,H*up,W*up).
    """
    def __init__(self, in_ch: int, up: int):
        super().__init__()
        self.up = up
        # a tiny head is enough; feel free to make this deeper if you like
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, (up * up) * 9, 1)
        )

    def forward(self, flow_lr: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        flow_lr: (B,2,H,W), feat: (B,C,H,W) at the SAME (H,W)
        returns flow_hr: (B,2,H*up,W*up)
        """
        B, _, H, W = flow_lr.shape
        up = self.up

        # predict mask and normalize over the 3x3 neighborhood
        mask = self.mask_head(feat)                              # (B, 9*up*up, H, W)
        mask = mask.view(B, 1, 9, up, up, H, W)
        mask = torch.softmax(mask, dim=2)                        # softmax across 9 neighbors

        # extract 3x3 patches from low-res flow
        flow_unf = F.unfold(flow_lr, kernel_size=3, padding=1)   # (B, 2*9, H*W)
        flow_unf = flow_unf.view(B, 2, 9, 1, 1, H, W)            # (B, 2, 9, 1, 1, H, W)

        # convex combination
        up_flow = torch.sum(mask * flow_unf, dim=2)              # (B, 2, up, up, H, W)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous() # (B, 2, H, up, W, up)
        up_flow = up_flow.view(B, 2, H * up, W * up)             # (B, 2, H*up, W*up)

        # IMPORTANT: scale by up to preserve flow units
        up_flow = up_flow * up
        return up_flow

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
        convex_up: int = 16,
        refinement_layers: int = 1,    # feature-scale refinement
    ):
        super().__init__()
        self.out_size = out_size
        self.radius = radius

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

        self.convex_upsampler = ConvexUpsampler(fusion_channels, up=convex_up)


    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        assert feat1.shape == feat2.shape, "feat1 and feat2 must have same shape"
        B, C, H, W = feat1.shape
        out_h, out_w = self.out_size

        f1 = F.silu(self.bn1(self.proj1(feat1)))
        f2 = F.silu(self.bn2(self.proj2(feat2)))

        corr = local_correlation(f1, f2, self.radius)
        diff = torch.abs(f1 - f2)
        x = torch.cat([f1, f2, diff, corr], dim=1)
        x = self.fuse_in(x)
        x = self.fuse_trunk(x)

        # coarse flow at feature scale
        flow = self.flow_head(x)

        # optional refinement at feature scale (still low-res)
        if self.refine is not None:
            flow = flow + self.refine(torch.cat([x, flow], dim=1))

        # Upsample with convex upsampler
        flow = self.convex_upsampler(flow, x)  # now ~16x; matches out_h/out_w if multiples

        # if output size differs a bit, small resize with correct scaling
        out_h, out_w = self.out_size
        B, _, Hf, Wf = flow.shape
        if (Hf, Wf) != (out_h, out_w):
            sx, sy = out_w / Wf, out_h / Hf
            flow[:,0] *= sx
            flow[:,1] *= sy
            flow = F.interpolate(flow, size=(out_h, out_w), mode='bilinear', align_corners=False)

        return flow
    
if __name__ == "__main__":
    B, C, H, W = 4, 384, 40, 40
    img_size = (640, 640)
    f1 = torch.randn(B, C, H, W).to("cuda")  # DINOv3 feat at t
    f2 = torch.randn(B, C, H, W).to("cuda")  # DINOv3 feat at t+1
    
    of_head = LiteFlowHead(out_size = (640, 640),
        in_channels= 384,
        proj_channels = 256,
        radius= 4,
        fusion_channels = 320,
        fusion_layers = 3,
        convex_up = 16,
        refinement_layers = 2).to("cuda")

    # ----------------- Utility: parameter counting -----------------
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print parameter counts
    print('Optical flow params: ', count_parameters(of_head))


    flow_img = of_head(f1, f2)
    print(flow_img.shape)