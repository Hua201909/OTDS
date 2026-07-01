train_eval.py (English Version)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .metrics import mae, smape
def run_experiment(model, train_loader, val_loader, test_loader,
                   loss_func, optimizer_config, device,
                   epochs=100, patience=20, model_save_path="best_model.pth"):
    """
    General training, validation and testing pipeline
    :param model: PyTorch model
    :param train_loader: Training DataLoader
    :param val_loader: Validation DataLoader
    :param test_loader: Testing DataLoader
    :param loss_func: Loss function with signature loss_func(model, inputs, targets) -> loss
    :param optimizer_config: Dictionary containing 'lr' and 'weight_decay'
    :param device: torch.device
    :param epochs: Maximum training epochs
    :param patience: Patience value for early stopping
 :param model_save_path: Save path for the best model
    :return: Dictionary containing training history and test metrics
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), **optimizer_config)
    base_criterion = nn.MSELoss()   # Used for standard MSE calculation during validation/testing
    train_losses = []
    val_mse_list = []
    val_rmse_list = []
    val_mae_list = []
    val_smape_list = []
    best_val_mse = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_func(model, inputs, labels)
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        # ---------- Validation ----------
        model.eval()
        val_sq_sum = 0.0
        val_mae_sum = 0.0
        val_smape_sum = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                batch_mse = base_criterion(outputs, labels).item()
                val_sq_sum += batch_mse * inputs.size(0)
                val_mae_sum += mae(outputs, labels).item() * inputs.size(0)
                val_smape_sum += smape(outputs, labels).item() * inputs.size(0)
        num_val = len(val_loader.dataset)
        epoch_val_mse = val_sq_sum / num_val
        epoch_val_rmse = torch.sqrt(torch.tensor(epoch_val_mse)).item()
        epoch_val_mae = val_mae_sum / num_val
        epoch_val_smape = val_smape_sum / num_val
 val_mse_list.append(epoch_val_mse)
        val_rmse_list.append(epoch_val_rmse)
        val_mae_list.append(epoch_val_mae)
        val_smape_list.append(epoch_val_smape)
 print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_train_loss:.6f}, "
              f"Val MSE: {epoch_val_mse:.6f}, RMSE: {epoch_val_rmse:.6f}, "
              f"MAE: {epoch_val_mae:.6f}, SMAPE: {epoch_val_smape:.2f}%")
        # Early stopping and model saving
        if epoch_val_mse < best_val_mse:
            best_val_mse = epoch_val_mse
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            print(f"  - Best model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No improvement for {patience} consecutive epochs.")
                break
    # ---------- Testing ----------
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_sq_sum = 0.0
    test_mae_sum = 0.0
    test_smape_sum = 0.0
    preds_scaled = []
    actuals_scaled = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            batch_mse = base_criterion(outputs, labels).item()
            test_sq_sum += batch_mse * inputs.size(0)
            test_mae_sum += mae(outputs, labels).item() * inputs.size(0)
            test_smape_sum += smape(outputs, labels).item() * inputs.size(0)
            preds_scaled.extend(outputs.cpu().numpy())
            actuals_scaled.extend(labels.cpu().numpy())
    num_test = len(test_loader.dataset)
    test_metrics = {
        "MSE": test_sq_sum / num_test,
        "RMSE": torch.sqrt(torch.tensor(test_sq_sum / num_test)).item(),
        "MAE": test_mae_sum / num_test,
        "SMAPE": test_smape_sum / num_test
    }
    history = {
        "train_losses": train_losses,
        "val_mse": val_mse_list,
        "val_rmse": val_rmse_list,
        "val_mae": val_mae_list,
        "val_smape": val_smape_list,}

      return { "history": history,"test_metrics": test_metrics,
        "best_model_path": model_save_path,
        "predictions_scaled": preds_scaled,
        "actuals_scaled": actuals_scaled }