import torch
import torch.nn as nn

def sam_fgsm_loss(model, inputs, targets, base_criterion=None,
                  epsilon_sam=0.001, lambda_sam=0.01,
                  epsilon_fgsm=0.001, lambda_fgsm=0.01):
    """
    Total loss = Original loss + λ_sam * SAM perturbation loss + λ_fgsm * FGSM adversarial loss
    """
    if base_criterion is None:
        base_criterion = nn.MSELoss()
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)

    # ---------------------- SAM 部分 ----------------------
    outputs = model(inputs)
    base_loss = base_criterion(outputs.squeeze(), targets)
    base_loss.backward(retain_graph=True)

    with torch.no_grad():
        original_params = {name: param.data.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.data = original_params[name] + epsilon_sam * torch.sign(param.grad.data)

        perturbed_outputs = model(inputs)
        sam_loss_val = base_criterion(perturbed_outputs.squeeze(), targets)

        # Restore parameters (to avoid affecting the subsequent FGSM calculation)
        for name, param in model.named_parameters():
            param.data = original_params[name]

    # ---------------------- The FGSM section ----------------------
    inputs.requires_grad_(True)
    outputs_fgsm = model(inputs)
    loss_fgsm = base_criterion(outputs_fgsm.squeeze(), targets)
    grad_x = torch.autograd.grad(loss_fgsm, inputs, retain_graph=True)[0]

    delta = epsilon_fgsm * torch.sign(grad_x)
    x_adv = inputs + delta
    x_adv = torch.clamp(x_adv, -5.0, 5.0)
    x_adv = x_adv.detach()

    outputs_adv = model(x_adv)
    fgsm_loss_val = base_criterion(outputs_adv.squeeze(), targets)

    # ---------------------- Total loss ----------------------
    total_loss = base_loss + lambda_sam * sam_loss_val + lambda_fgsm * fgsm_loss_val
    return total_loss.mean()