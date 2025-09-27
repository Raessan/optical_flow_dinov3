import torch
import torch.nn.functional as F

def warp_with_flow(img1, flow):
    """
    img1: [1,3,H,W] or [B,3,H,W] in float (0..1 or 0..255)
    flow: [1,2,H,W]             forward flow in pixels (u,v)
    returns: warped img1 -> img1_warped sampled at (x+u, y+v)
    """
    B, C, H, W = img1.shape
    # base grid in pixel coords
    y, x = torch.meshgrid(torch.arange(H, device=img1.device),
                          torch.arange(W, device=img1.device), indexing="ij")
    x = x.unsqueeze(0).expand(B, -1, -1).float()
    y = y.unsqueeze(0).expand(B, -1, -1).float()
    u, v = flow[:, 0], flow[:, 1]
    x2 = x + u
    y2 = y + v

    # normalize to [-1,1] for grid_sample
    x2n = 2.0 * (x2 / (W - 1)) - 1.0
    y2n = 2.0 * (y2 / (H - 1)) - 1.0
    grid = torch.stack([x2n, y2n], dim=-1)  # [B,H,W,2]

    img1_w = F.grid_sample(img1, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # validity mask: inside image bounds
    in_bounds = (x2 >= 0) & (x2 <= W - 1) & (y2 >= 0) & (y2 <= H - 1)
    in_bounds = in_bounds.unsqueeze(1).float()  # [B,1,H,W]
    return img1_w, in_bounds

def warp_img2_to_img1(img2, flow12):
    """
    img2:  [B,3,H,W]  next frame (t+1)
    flow12:[B,2,H,W]  forward flow img1->img2 in pixels (u,v)
    returns: img2 warped into img1's frame, and a valid mask
    """
    B, C, H, W = img2.shape
    # base grid in pixel coords
    y, x = torch.meshgrid(
        torch.arange(H, device=img2.device),
        torch.arange(W, device=img2.device),
        indexing="ij"
    )
    x = x.float().unsqueeze(0).expand(B, -1, -1)
    y = y.float().unsqueeze(0).expand(B, -1, -1)

    u, v = flow12[:, 0], flow12[:, 1]
    xs = x + u   # where to sample img2 horizontally
    ys = y + v   # where to sample img2 vertically

    # normalize to [-1,1] for grid_sample with align_corners=True
    xs_n = 2.0 * (xs / (W - 1)) - 1.0
    ys_n = 2.0 * (ys / (H - 1)) - 1.0
    grid = torch.stack([xs_n, ys_n], dim=-1)  # [B,H,W,2]

    img2_warped = F.grid_sample(
        img2, grid, mode="bilinear",
        padding_mode="zeros", align_corners=True
    )

    # in-bounds validity mask
    valid = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    valid = valid.unsqueeze(1).float()
    return img2_warped, valid

def photometric_check(img1, img2, flow, valid_mask=None):
    """
    Returns average L1 error on valid, in-bounds pixels and the coverage.
    img*: [3,H,W] or [1,3,H,W], flow: [2,H,W] or [1,2,H,W]
    valid_mask: optional [1,1,H,W] (e.g., from sparse GT or padding mask)
    """
    if img1.dim() == 3: img1 = img1.unsqueeze(0)
    if img2.dim() == 3: img2 = img2.unsqueeze(0)
    if flow.dim() == 3: flow = flow.unsqueeze(0)

    # Cast types
    img1 = img1.to(torch.float32)
    flow = flow.to(torch.float32)

    img1_w, inb = warp_with_flow(img1, flow)

    # If images are 0..255, normalize for error
    if img1_w.max() > 1.5 or img2.max() > 1.5:
        img1_w = img1_w / 255.0
        img2 = img2 / 255.0

    mask = inb
    if valid_mask is not None:
        mask = mask * (valid_mask > 0).float()

    denom = mask.sum().clamp(min=1.0)
    l1 = (mask * (img1_w - img2).abs()).sum() / denom
    coverage = (mask.mean()).item()
    return l1.item(), coverage, img1_w

def photometric_check_v2(img1, img2, flow, valid_mask=None):
    """
    Returns average L1 error on valid, in-bounds pixels and the coverage.
    img*: [3,H,W] or [1,3,H,W], flow: [2,H,W] or [1,2,H,W]
    valid_mask: optional [1,1,H,W] (e.g., from sparse GT or padding mask)
    """
    if img1.dim() == 3: img1 = img1.unsqueeze(0)
    if img2.dim() == 3: img2 = img2.unsqueeze(0)
    if flow.dim() == 3: flow = flow.unsqueeze(0)

    # Cast types
    img2 = img2.to(torch.float32)
    flow = flow.to(torch.float32)

    img2_w, inb = warp_img2_to_img1(img2, flow)

    # If images are 0..255, normalize for error
    if img2_w.max() > 1.5 or img2.max() > 1.5:
        img2_w = img2_w / 255.0
        img1 = img1 / 255.0

    mask = inb
    if valid_mask is not None:
        mask = mask * (valid_mask > 0).float()

    denom = mask.sum().clamp(min=1.0)
    l1 = (mask * (img2_w - img1).abs()).sum() / denom
    coverage = (mask.mean()).item()
    return l1.item(), coverage, img2_w