# src/drl/validation.py

import os
import torch
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
