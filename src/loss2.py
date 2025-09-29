import torch
import torch.nn.functional as F

def charbonnier(x, eps=1e-3):
    # robust L1: sqrt(x^2 + eps^2)
    return torch.sqrt(x * x + eps * eps)

def image_to_gray(img):
    # img: (B,3,H,W) in [0,1] or [0,255]
    if img.shape[1] == 1:
        return img
    r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    return gray

def sobel_grad(x):
    # x: (B,1,H,W) or (B,C,H,W); returns gx, gy same shape
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    C = x.shape[1]
    kx = kx.repeat(C,1,1,1)
    ky = ky.repeat(C,1,1,1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy

def flow_gradients(flow):
    # flow: (B,2,H,W)
    fx, fy = flow[:,0:1], flow[:,1:2]
    fx_gx, fx_gy = sobel_grad(fx)
    fy_gx, fy_gy = sobel_grad(fy)
    # stack channel-wise: (B,2,H,W) for each direction
    gx = torch.cat([fx_gx, fy_gx], dim=1)
    gy = torch.cat([fx_gy, fy_gy], dim=1)
    return gx, gy

def ssim(x, y, C1=0.01**2, C2=0.03**2):  # simple, window-free SSIM approx
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x*mu_x
    sigma_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
    ssim_n = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    ssim_d = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return torch.clamp((1 - ssim_map) / 2, 0, 1)  # DSSIM in [0,1]

def warp(img, flow):
    # img: (B,C,H,W), flow: (B,2,H,W) in pixels (u,v)
    B, C, H, W = img.shape
    # build base grid in [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype),
        torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy], dim=-1)  # (H,W,2)
    # convert pixel flow to normalized coords
    u = flow[:,0] / ((W-1)/2)
    v = flow[:,1] / ((H-1)/2)
    grid = base_grid[None].repeat(B,1,1,1)
    grid = torch.stack([grid[...,0] + u, grid[...,1] + v], dim=-1)  # (B,H,W,2)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

def sequence_loss(
    flow_preds, flow_gt, valid, img1=None, img2=None, *,
    gamma=0.8, max_flow=400,
    w_epe=1.0, w_smooth=1.5, w_grad=0.5, w_photo=0.0,
    edge_alpha=4.0, smooth_eps=1e-3
):
    """
    A sharpening loss:
      - robust Charbonnier EPE over (possibly) a sequence of predictions
      - edge-aware smoothness (weights from image gradients)
      - gradient matching vs GT near edges
      - optional photometric (SSIM + Charbonnier) on I2 warped to I1
    """
    if isinstance(flow_preds, torch.Tensor):
        flow_preds = [flow_preds]

    n_predictions = len(flow_preds)

    # validity mask
    mag = torch.sqrt(torch.sum(flow_gt**2, dim=1))
    valid_mask = (valid >= 0.5) & (mag < max_flow)  # (B,H,W)
    V = valid_mask[:, None].float()  # (B,1,H,W)

    # ---------- robust EPE over sequence ----------
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        epe_i = torch.sqrt(torch.sum((flow_preds[i] - flow_gt)**2, dim=1, keepdim=True))  # (B,1,H,W)
        loss_i = charbonnier(epe_i) * V
        flow_loss += i_weight * loss_i.mean()

    # ---------- edge-aware smoothness ----------
    # need guidance from img1 for edge weights
    if img1 is not None:
        g = image_to_gray(img1)
        gx_I, gy_I = sobel_grad(g)          # (B,1,H,W)
        edge_w = torch.exp(-edge_alpha * torch.sqrt(gx_I**2 + gy_I**2))  # low weight on edges
    else:
        edge_w = 1.0

    # smoothness on the *final* prediction only
    flow_final = flow_preds[-1]
    gx_F, gy_F = flow_gradients(flow_final)  # each: (B,2,H,W)
    smooth = (charbonnier(gx_F, smooth_eps) + charbonnier(gy_F, smooth_eps))
    if isinstance(edge_w, torch.Tensor):
        smooth = smooth * edge_w  # broadcast to (B,2,H,W)
    smooth = (smooth * V).mean()

    # ---------- gradient matching vs GT (sharp boundaries) ----------
    gx_GT, gy_GT = flow_gradients(flow_gt)
    grad_diff = charbonnier(gx_F - gx_GT) + charbonnier(gy_F - gy_GT)
    # emphasize where image has edges (1 - edge_w)
    if isinstance(edge_w, torch.Tensor):
        edge_emph = (1.0 - edge_w).clamp(0, 1)
        grad_diff = grad_diff * edge_emph
    grad_diff = (grad_diff * V).mean()

    # ---------- optional photometric consistency ----------
    photo = flow_final.new_tensor(0.0)
    if w_photo > 0 and (img1 is not None and img2 is not None):
        I1 = img1
        I2w = warp(img2, flow_final)
        # DSSIM + Charbonnier on residual
        dssim = ssim(I1, I2w).mean(1, keepdim=True)   # per-pixel, avg over channels
        l1_ph = charbonnier(I1 - I2w).mean(1, keepdim=True)
        photometric = (0.85 * dssim + 0.15 * l1_ph)
        photo = (photometric * V).mean()

    total = w_epe*flow_loss + w_smooth*smooth + w_grad*grad_diff + w_photo*photo

    # ---------- metrics ----------
    epe = torch.sqrt(torch.sum((flow_final - flow_gt)**2, dim=1))  # (B,H,W)
    epe = epe.view(-1)[valid_mask.view(-1)]
    metrics = {
        'epe': epe.mean().item() if epe.numel() else 0.0,
        '1px': (epe < 1).float().mean().item() if epe.numel() else 0.0,
        '3px': (epe < 3).float().mean().item() if epe.numel() else 0.0,
        '5px': (epe < 5).float().mean().item() if epe.numel() else 0.0,
        'L_epe': (w_epe*flow_loss).item(),
        'L_smooth': (w_smooth*smooth).item(),
        'L_grad': (w_grad*grad_diff).item(),
        'L_photo': (w_photo*photo).item() if w_photo > 0 else 0.0,
        'total': total.item(),
    }
    return total, w_epe*flow_loss, w_smooth*smooth, w_grad*grad_diff, w_photo*photo, metrics

if __name__ == "__main__":
    B, H, W = 2, 100, 100   # batch=2, 3x3 flow fields

    # Ground truth flow: all ones
    flow_gt = torch.ones(B, 2, H, W)

    # Predictions: one close to gt, one far from gt
    flow_pred1 = torch.ones(B, 2, H, W) * 1.2
    flow_pred2 = torch.ones(B, 2, H, W) * 1.0
    flow_preds = [flow_pred1, flow_pred2]

    # All pixels valid
    valid = torch.ones(B, H, W)

    loss, flow, smooth, grad, photo, metrics = sequence_loss(flow_preds, flow_gt, valid)

    print("Loss:", loss.item())
    print("Metrics:", metrics)
