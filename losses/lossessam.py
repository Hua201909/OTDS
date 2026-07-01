import torch
import torch.nn as nn

def sam_loss(model, inputs, targets, base_criterion=None,
             epsilon_sam=0.001, lambda_sam=0.01):
    """
    Total loss = Original loss + λ_sam * Parameter perturbation loss
    """
    if base_criterion is None:
        base_criterion = nn.MSELoss()
    device = next(model.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)

    # 1. Original loss + Gradient obtained through backpropagation
    outputs = model(inputs)
    base_loss = base_criterion(outputs.squeeze(), targets)
    base_loss.backward(retain_graph=True)

    # 2. parameter perturbation
    with torch.no_grad():
        original_params = {name: param.data.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.data = original_params[name] + epsilon_sam * torch.sign(param.grad.data)

        perturbed_outputs = model(inputs)
        perturbed_loss = base_criterion(perturbed_outputs.squeeze(), targets)

        # Restore original parameters
        for name, param in model.named_parameters():
            param.data = original_params[name]

    total_loss = base_loss + lambda_sam * perturbed_loss
    return total_loss.mean()