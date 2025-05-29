#validate_deep_router.py

import os
import torch
from torch.utils.data import DataLoader

from drl.deep_dataset import DeepRoutingDataset
from drl.deep_router import DeepRoutingUNet
from drl.validation import validation_predictions
from drl.utils.config_loader import load_config

def validate_deep_model(config_path='config/config.yaml', model_filename='deep_routing_model.pth'):
    # Load config
    cfg = load_config(config_path)
    dataset_cfg = cfg['dataset']
    train_cfg = cfg['training']
    device = train_cfg['device']

    # Dataset and DataLoader
    val_dataset = DeepRoutingDataset(cfg, split='val')
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    # Reconstruct model architecture
    in_channels = 1 + dataset_cfg['rain_snapshots']
    model = DeepRoutingUNet(in_channels=in_channels, out_channels=1)
    model = model.to(device)

    # Load saved weights
    model_path = os.path.join(train_cfg['model_dir'], model_filename)
    assert os.path.exists(model_path), f"❌ Model not found: {model_path}"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded model from {model_path}")

    # Run validation
    validation_predictions(model, val_loader, device)

if __name__ == "__main__":
    print("🔍 Running validation on saved deep routing model...")
    validate_deep_model()
    print("Validation plots should now be available")
