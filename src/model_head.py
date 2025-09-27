import torch
import torch.nn as nn
import torch.nn.functional as F

# --- helpers ---

def normalize_feat(x, eps=1e-6):
    # L2-normalize channel-wise to make correlation behave
    return x / (x.norm(dim=1, keepdim=True) + eps)

def local_correlation(f1, f2, radius=4):
    """
    f1,f2: [B,C,H,W] (assumed L2-normalized)
    returns: cost volume [B,(2r+1)^2,H,W] with local correlations
    """
    B, C, H, W = f1.shape
    r = radius
    # pad f2 for local shifts
    f2p = F.pad(f2, (r, r, r, r))
    vols = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            # extract shifted f2 (crop window)
            ys, ye = dy + r, dy + r + H
            xs, xe = dx + r, dx + r + W
            f2s = f2p[:, :, ys:ye, xs:xe]
            vols.append((f1 * f2s).sum(1, keepdim=True))  # inner product across C
    vol = torch.cat(vols, dim=1) / (C**0.5)  # [B,(2r+1)^2,H,W]
    return vol

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.conv(x)))

# --- the head ---

class Dinov3FlowHead(nn.Module):
    """
    Lightweight optical flow head for DINOv3 features.
    Input:  f1, f2 = [B,384,40,40]  (two frames’ features)
    Output: flow at feature res [B,2,H,W] in *feature pixels*,
            and optionally upsampled flow in image pixels.
    """
    def __init__(self, img_size, in_ch=384, radius=4, hidden=128, add_coords=True):
        super().__init__()
        self.img_size = img_size
        self.radius = radius
        self.add_coords = add_coords
        corr_ch = (2*radius + 1) ** 2
        coord_ch = 2 if add_coords else 0
        head_in = corr_ch + in_ch + coord_ch

        self.stem = nn.Sequential(
            ConvBlock(head_in, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
        )
        # a tiny refinement with residual blocks
        self.refine = nn.Sequential(
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
        )
        self.pred = nn.Conv2d(hidden, 2, kernel_size=3, padding=1)

    @staticmethod
    def _make_coords(B,H,W,device):
        y,x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        x = x.float().unsqueeze(0).expand(B,-1,-1)
        y = y.float().unsqueeze(0).expand(B,-1,-1)
        return torch.stack([x,y], dim=1)  # [B,2,H,W]

    def forward(self, f1, f2):
        """
        f1,f2: [B,C,H,W] DINOv3 features for frame t and t+1
        Returns:
          flow_feat: [B,2,H,W] in *feature* pixels
          flow_img: [B,2,H_img,W_img] in *image* pixels
        """
        B, C, H, W = f1.shape
        f1n = normalize_feat(f1)
        f2n = normalize_feat(f2)

        corr = local_correlation(f1n, f2n, radius=self.radius)  # [B,K,H,W]
        feats = [corr, f1]  # keep raw f1 to give appearance cues

        if self.add_coords:
            coords = self._make_coords(B,H,W,f1.device)  # helps network know position
            feats.append(coords)

        x = torch.cat(feats, dim=1)
        x = self.stem(x) + 0  # stem features
        x = self.refine(x) + x  # residual
        flow_feat = self.pred(x)  # [B,2,H,W], units = feature pixels

        H_img, W_img = self.img_size
        # scale factor from feature map to image
        sy = H_img / float(H)
        sx = W_img / float(W)
        flow_img = F.interpolate(flow_feat, size=(H_img, W_img), mode='bilinear', align_corners=True)
        flow_img[:,0] *= sx
        flow_img[:,1] *= sy
        return flow_feat, flow_img
    

if __name__ == "__main__":
    B, C, H, W = 3, 384, 40, 40
    img_size = (640, 640)
    f1 = torch.randn(B, C, H, W)  # DINOv3 feat at t
    f2 = torch.randn(B, C, H, W)  # DINOv3 feat at t+1

    of_head = Dinov3FlowHead(img_size, in_ch=C, radius=4, hidden=128, add_coords=True)

    # ----------------- Utility: parameter counting -----------------
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print parameter counts
    print('Depth params: ', count_parameters(of_head))


    flow_feat, flow_img = of_head(f1, f2)  # if your image is 320×320
    print(flow_img.shape)