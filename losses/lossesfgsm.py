import torch
import torch.nn as nn

def fgsm_loss(model, inputs, targets, base_criterion=None,
              epsilon_fgsm=0.001, lambda_fgsm=0.01):
    """
    Total loss = original loss + λ_fgsm * FGSM adversarial loss
    """
    if base_criterion is None:
        base_criterion = nn.MSELoss()
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)

    # 1. original loss
    outputs = model(inputs)
    base_loss = base_criterion(outputs.squeeze(), targets)

    # 2. FGSM adversarial loss
    inputs.requires_grad_(True)
    outputs_fgsm = model(inputs)
    loss_fgsm = base_criterion(outputs_fgsm.squeeze(), targets)
    grad_x = torch.autograd.grad(loss_fgsm, inputs, retain_graph=True)[0]

    delta = epsilon_fgsm * torch.sign(grad_x)
    x_adv = inputs + delta
    x_adv = torch.clamp(x_adv, -5.0, 5.0)   # 裁剪到标准化后的合理范围
    x_adv = x_adv.detach()

    outputs_adv = model(x_adv)
    fgsm_loss_val = base_criterion(outputs_adv.squeeze(), targets)

    total_loss = base_loss + lambda_fgsm * fgsm_loss_val
    return total_loss.mean()