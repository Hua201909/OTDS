import torch.nn as nn

def baseline_loss(model, inputs, targets, base_criterion=None):
    """
    Standard training loss: using only the basic MSE loss.
    """
    if base_criterion is None:
        base_criterion = nn.MSELoss()
    outputs = model(inputs).squeeze()
    loss = base_criterion(outputs, targets)
    return loss