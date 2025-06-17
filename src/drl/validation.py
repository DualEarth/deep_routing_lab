# src/drl/validation.py

import os
import torch
import torch.nn as nn
from drl.utils.viz import plot_comparison
from drl.utils.tensor_ops import center_crop_to_match
import matplotlib.image as mpimg

def validation_predictions(model, 
                           val_loader, 
                           device,
                           png_dir,
                           save_dir
                           ):
    """
    Run the model on the validation loader, load each sample's DEM,
    and call plot_comparison(y_true, y_pred, dem, index, save_dir).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    batch_size = val_loader.batch_size

    with torch.no_grad():
        for i, (x, y_true) in enumerate(val_loader):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            y_pred = center_crop_to_match(y_pred, y_true)

            for j in range(x.size(0)):
                sample_id = i * batch_size + j
                sample_dir = os.path.join(png_dir, f"sample_{sample_id:05d}")
                dem_path   = os.path.join(sample_dir, "dem.png")

                # load DEM as 2D array
                dem = mpimg.imread(dem_path)
                if dem.ndim == 3:  
                    dem = dem[..., 0]  # drop color channels if present

                y_true_np = y_true[j].cpu().numpy()
                y_pred_np = y_pred[j].cpu().numpy()

                plot_comparison(
                    y_true_np,
                    y_pred_np,
                    dem,
                    index=sample_id,
                    save_dir=save_dir
                )

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