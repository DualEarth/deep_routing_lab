import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from drl.deep_dataset import DeepRoutingDataset
from drl.deep_router import DeepRoutingUNet
from drl.validation import validation_predictions
from drl.utils.config_loader import load_config
from drl.utils.tensor_ops import center_crop_to_match
from drl.utils.viz import plot_comparison

def train_deep_model(config_path='config/config.yaml'):
    cfg = load_config(config_path)
    dataset_cfg = cfg['dataset']
    train_cfg = cfg['training']

    print(dataset_cfg)
    print(train_cfg)

    # Dataset and DataLoader
    train_dataset = DeepRoutingDataset(cfg, split='train')
    val_dataset = DeepRoutingDataset(cfg, split='val')
    print(val_dataset)

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)
    print(train_loader)

    print(val_loader)

    # Model
    in_channels = 1 + dataset_cfg['rain_snapshots']
    model = DeepRoutingUNet(in_channels=in_channels, out_channels=1)
    print(model)

    model = model.to(train_cfg['device'])
    print(next(model.parameters()).device)

    # # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg['learning_rate']))
    loss_fn = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(train_cfg['epochs']):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")

        for x, y in pbar:
            x = x.to(train_cfg['device'])
            y = y.to(train_cfg['device'])

            optimizer.zero_grad()
            y_pred = model(x)

            # Align prediction shape to target
            y_pred = center_crop_to_match(y_pred, y)
            
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch+1}: Avg Loss = {epoch_loss / len(train_loader):.6f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(train_cfg['device'])
                y_val = y_val.to(train_cfg['device'])
                y_pred_val = model(x_val)
                y_pred_val = center_crop_to_match(y_pred_val, y_val)
                val_loss += loss_fn(y_pred_val, y_val).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss = {avg_val_loss:.6f}")
        model.train()
        
        os.makedirs(train_cfg['model_dir'], exist_ok=True)
        model_path = os.path.join(train_cfg['model_dir'], 'deep_routing_model.pth')
        torch.save(model.state_dict(), model_path)

    return model, val_loader, train_cfg['device']

if __name__ == "__main__":
    print("training deep router")
    model, val_loader, device = train_deep_model()
    validation_predictions(model, val_loader, device)
