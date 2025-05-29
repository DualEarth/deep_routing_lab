# src/drl/validation.py

import os
import torch
import torch.nn as nn
from drl.utils.viz import plot_comparison
from drl.utils.tensor_ops import center_crop_to_match

def validation_predictions(model, val_loader, device, save_dir="validation_plots"):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, (x, y_true) in enumerate(val_loader):
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model(x)
            y_pred = center_crop_to_match(y_pred, y_true)

            for j in range(x.size(0)):  # iterate through batch
                y_true_np = y_true[j].cpu().numpy()
                y_pred_np = y_pred[j].cpu().numpy()
                plot_comparison(y_true_np, y_pred_np, index=i * x.size(0) + j, save_dir=save_dir)

def compute_validation_loss(model, val_loader, device, loss_fn=None):
    """
    Compute average validation loss over the dataset.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            y_pred_val = model(x_val)
            y_pred_val = center_crop_to_match(y_pred_val, y_val)

            val_loss += loss_fn(y_pred_val, y_val).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss = {avg_val_loss:.6f}")
    model.train()
    return avg_val_loss