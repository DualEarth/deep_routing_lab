import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from drl.deep_dataset import DeepRoutingDataset
from drl.deep_router import DeepRoutingUNet
from drl.utils.config_loader import load_config
from drl.utils.tensor_ops import center_crop_to_match

def train_deep_model(config_path='config/config.yaml'):
    cfg = load_config(config_path)
    dataset_cfg = cfg['dataset']
    train_cfg = cfg['training']

    print(dataset_cfg)
    print(train_cfg)

    # Dataset and DataLoader
    train_dataset = DeepRoutingDataset(
        root_dir=dataset_cfg['png_dir'],
        num_rain_frames=dataset_cfg['rain_snapshots'],
        apply_cloud_mask=dataset_cfg.get('apply_cloud_mask', False),
        cloud_mask_fn=None  # Replace with actual function if desired
    )
    print(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    print(train_loader)

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

    # # Save model
    # os.makedirs(train_cfg['model_dir'], exist_ok=True)
    # model_path = os.path.join(train_cfg['model_dir'], 'deep_model.pth')
    # torch.save(model.state_dict(), model_path)
    # print(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    train_deep_model()
