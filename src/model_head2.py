import math
from typing import Tuple, Optional
import re

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
    """Squeeze-and-Excitation (channel attention), tiny and effective."""
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class DSConvBlock(nn.Module):
    """DS conv + SE, repeated N times."""
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
    Compute correlation volume within a (2r+1)^2 neighborhood.
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
# RAFT-style convex upsampler
# ----------------------------

class ConvexUpsampler(nn.Module):
    """
    Learned convex upsampler (à la RAFT).
    Predicts a mask (softmax over 3x3 neighbors) for each subpixel in an s×s grid,
    then blends a 3x3 neighborhood of the coarse flow and reshapes to (H*s, W*s).
    """
    def __init__(self, in_ch: int, scale: int):
        super().__init__()
        assert scale >= 2 and isinstance(scale, int), "Convex upsampler needs integer scale >= 2"
        self.scale = scale
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, 9 * scale * scale, 1)
        )

    def forward(self, feat: torch.Tensor, flow_coarse: torch.Tensor) -> torch.Tensor:
        # feat: (B, in_ch, H, W); flow_coarse: (B, 2, H, W)
        s = self.scale
        B, _, H, W = flow_coarse.shape

        # Predict per-subpixel convex weights over 3x3 neighborhood
        mask = self.mask_head(feat)                       # (B, 9*s*s, H, W)
        mask = mask.view(B, 1, 9, s, s, H, W)             # (B, 1, 9, s, s, H, W)
        mask = torch.softmax(mask, dim=2)                 # softmax over 9 neighbors

        # Collect 3x3 neighbors of coarse flow
        unfold = F.unfold(flow_coarse, kernel_size=3, padding=1)  # (B, 2*9, H*W)
        unfold = unfold.view(B, 2, 9, 1, 1, H, W)                  # (B, 2, 9, 1, 1, H, W)

        # Blend and reshape to high-res grid
        up_flow = (mask * unfold).sum(dim=2)        # (B, 2, 1, s, s, H, W)
        up_flow = up_flow.squeeze(2)                 # (B, 2, s, s, H, W)
        # reorder (B,2,s,s,H,W) -> (B,2,H,s,W,s)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # (B, 2, H, s, W, s)
        up_flow = up_flow.reshape(B, 2, H * s, W * s)
        # scale vectors (flow defined on low-res grid)
        up_flow[:, 0] *= s
        up_flow[:, 1] *= s
        return up_flow

# ----------------------------
# Flow Head
# ----------------------------

class LiteFlowHead(nn.Module):
    """
    Lightweight, expressive optical-flow head for features from two frames.
    - Inputs: feat1, feat2 of shape (B, C, H, W)
    - Output: flow at size (B, 2, out_h, out_w)
    """
    def __init__(
        self,
        out_size: Tuple[int, int] = (640, 640),
        in_channels: int = 384,
        proj_channels: int = 128,
        radius: int = 4,                 # local search radius (2r+1)^2 cost channels
        fusion_channels: int = 256,      # width of fusion trunk
        fusion_layers: int = 2,          # depth in DSConv blocks
        refinement_layers: int = 1,      # extra refinement after initial flow
        use_convex_upsampling: bool = True,  # replaces pixel shuffle to avoid checkerboard
        enable_resize_refine: bool = True,   # refine after bilinear upsample (for non-integer or unmatched scales)
    ):
        super().__init__()
        self.out_size = out_size
        self.use_convex_upsampling = use_convex_upsampling
        self.enable_resize_refine = enable_resize_refine

        # Project backbone features
        self.proj1 = nn.Conv2d(in_channels, proj_channels, 1, bias=False)
        self.proj2 = nn.Conv2d(in_channels, proj_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(proj_channels)
        self.bn2 = nn.BatchNorm2d(proj_channels)

        self.radius = radius
        cv_ch = (2 * radius + 1) ** 2

        # Feature fusion: [f1, f2, |f1-f2|, corr]
        fuse_in = proj_channels * 3 + cv_ch
        self.fuse_in = nn.Sequential(
            DepthwiseSeparableConv(fuse_in, fusion_channels, k=3),
            SE(fusion_channels, r=8),
        )
        self.fuse_trunk = DSConvBlock(fusion_channels, hidden=fusion_channels, num_layers=fusion_layers)

        # Predict coarse flow at feature resolution
        self.flow_head = nn.Sequential(
            DepthwiseSeparableConv(fusion_channels, fusion_channels, k=3),
            nn.Conv2d(fusion_channels, 2, 1)
        )

        # Optional refinement on (features + coarse flow)
        if refinement_layers > 0:
            ref_in = fusion_channels + 2
            self.refine = nn.Sequential(
                DepthwiseSeparableConv(ref_in, fusion_channels, k=3),
                DSConvBlock(fusion_channels, num_layers=refinement_layers),
                nn.Conv2d(fusion_channels, 2, 1)
            )
        else:
            self.refine = None

        # Convex upsamplers cached per (scale, in_ch) with proper registration
        # Key format: "s{scale}_c{in_ch}"
        self.convex_ups = nn.ModuleDict()

        # Optional resize-then-refine (created lazily)
        self.refine_up = None

    @torch.no_grad()
    def _infer_out_size(self, feat: torch.Tensor, out_size: Optional[Tuple[int, int]]):
        B, C, H, W = feat.shape
        if out_size is None:
            # default: upscale to a typical backbone stride (e.g., 16x)
            return (H * 16, W * 16)
        return out_size

    def _get_convex_up(self, scale: int, in_ch: int, device: torch.device) -> nn.Module:
        key = f"s{scale}_c{in_ch}"
        if key not in self.convex_ups:
            # create on requested device
            self.convex_ups[key] = ConvexUpsampler(in_ch, scale).to(device)
        return self.convex_ups[key]

    def _resize_then_refine(self, x_feats: torch.Tensor, flow_lo: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        """
        Baseline, artifact-free path: bilinear upsample + small refinement on concatenated [feat_up, flow_up].
        """
        B, _, H, W = flow_lo.shape
        # Upsample flow vectors and scale properly
        flow_up = F.interpolate(flow_lo, size=(out_h, out_w), mode='bilinear', align_corners=False)
        flow_up[:, 0] *= (out_w / W)
        flow_up[:, 1] *= (out_h / H)

        if not self.enable_resize_refine:
            return flow_up

        if self.refine_up is None:
            in_ch = x_feats.shape[1] + 2
            self.refine_up = nn.Sequential(
                DepthwiseSeparableConv(in_ch, x_feats.shape[1], k=3),
                nn.Conv2d(x_feats.shape[1], 2, 1)
            ).to(x_feats.device)

        feat_up = F.interpolate(x_feats, size=(out_h, out_w), mode='bilinear', align_corners=False)
        flow_up = flow_up + self.refine_up(torch.cat([feat_up, flow_up], dim=1))
        return flow_up

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
    ) -> torch.Tensor:
        """
        feat1, feat2: (B, C, H, W)
        Returns: flow (B, 2, out_h, out_w)
        """
        assert feat1.shape == feat2.shape, "feat1 and feat2 must have same shape"
        B, C, H, W = feat1.shape
        out_h, out_w = self._infer_out_size(feat1, self.out_size)

        # Backbone projections
        f1 = F.silu(self.bn1(self.proj1(feat1)))
        f2 = F.silu(self.bn2(self.proj2(feat2)))

        # Local cost volume
        corr = local_correlation(f1, f2, self.radius)  # (B, KK, H, W)

        # Simple feature diffs
        diff = torch.abs(f1 - f2)

        # Fusion trunk
        x = torch.cat([f1, f2, diff, corr], dim=1)
        x = self.fuse_in(x)
        x = self.fuse_trunk(x)

        # Coarse flow at feature resolution
        flow_coarse = self.flow_head(x)

        # Optional refinement at feature scale
        if self.refine is not None:
            x_ref = torch.cat([x, flow_coarse], dim=1)
            delta = self.refine(x_ref)
            flow_coarse = flow_coarse + delta

        # Upsample to requested size
        scale_h = out_h / H
        scale_w = out_w / W
        can_use_integer_same_scale = float(scale_h).is_integer() and float(scale_w).is_integer() and int(scale_h) == int(scale_w)

        if self.use_convex_upsampling and can_use_integer_same_scale and int(scale_h) >= 2:
            s = int(scale_h)
            up_in = torch.cat([x, flow_coarse], dim=1)  # use features+flow for mask prediction
            convex_up = self._get_convex_up(s, up_in.shape[1], feat1.device)
            flow_up = convex_up(up_in, flow_coarse)     # already scaled by s inside
            # If requested out_size isn't exactly H*s/W*s (rare rounding), final safety resize + exact scaling adjust:
            if (flow_up.shape[-2] != out_h) or (flow_up.shape[-1] != out_w):
                # Resample and rescale vectors to exact size
                prev_h, prev_w = flow_up.shape[-2:]
                flow_up = F.interpolate(flow_up, size=(out_h, out_w), mode='bilinear', align_corners=False)
                flow_up[:, 0] *= (out_w / prev_w)
                flow_up[:, 1] *= (out_h / prev_h)
        else:
            # Artifact-free fallback
            flow_up = self._resize_then_refine(x, flow_coarse, out_h, out_w)

        return flow_up
    
if __name__ == "__main__":
    B, C, H, W = 3, 384, 40, 40
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