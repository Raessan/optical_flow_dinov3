import torch

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """

    # allow single prediction tensor
    if isinstance(flow_preds, torch.Tensor):
        flow_preds = [flow_preds]

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

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

    loss, metrics = sequence_loss(flow_preds, flow_gt, valid)

    print("Loss:", loss.item())
    print("Metrics:", metrics)
