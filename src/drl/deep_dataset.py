import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from drl.utils.tensor_ops import pad_to_multiple
from drl.utils.config_loader import load_config


def load_image(path):
    """Load an image file as a grayscale float32 numpy array normalized to [0, 1]."""
    return np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0


def stack_rain_images(sample_dir, num_rain_frames):
    rain_stack = []
    for i in range(num_rain_frames):
        rain_path = os.path.join(sample_dir, f"rain_{i:03d}.png")
        rain_img = load_image(rain_path)
        rain_stack.append(rain_img)
    return np.stack(rain_stack, axis=0)  # Shape: [C, H, W]


class DeepRoutingDataset(Dataset):
    def __init__(self, cfg, split='train'):
        dataset_cfg = cfg['dataset']
        training_cfg = cfg['training']

        self.root_dir = dataset_cfg['png_dir']
        all_samples = sorted([d for d in os.listdir(self.root_dir) if d.startswith("sample_")])
        n = len(all_samples)

        # Load split ratios from config
        split_ratios = training_cfg['split_ratios']
        train_end = int(split_ratios[0] * n)
        val_end = train_end + int(split_ratios[1] * n)

        if split == 'train':
            self.samples = all_samples[:train_end]
        elif split == 'val':
            self.samples = all_samples[train_end:val_end]
        elif split == 'test':
            self.samples = all_samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.num_rain_frames = dataset_cfg['rain_snapshots']
        self.apply_cloud_mask = training_cfg.get('apply_cloud_mask', False)
        self.cloud_mask_fn = None  # replace if needed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_dir = os.path.join(self.root_dir, sample_name)

        dem = load_image(os.path.join(sample_dir, "dem.png"))
        assert dem.ndim == 2, f"DEM shape wrong: {dem.shape}"
        h_sample = load_image(os.path.join(sample_dir, "h_sample.png"))
        rain_stack = stack_rain_images(sample_dir, self.num_rain_frames)

        if self.apply_cloud_mask and self.cloud_mask_fn is not None:
            mask = self.cloud_mask_fn(h_sample.shape)
            h_sample = np.where(mask == 1, 0.0, h_sample)

        assert h_sample.ndim == 2, f"h_sample shape wrong: {h_sample.shape}"
        assert rain_stack.ndim == 3, f"Rain stack shape wrong: {rain_stack.shape}"

        # load dem, h_sample, rain_stack as np arrays...
        dem = pad_to_multiple(dem, multiple=16)
        h_sample = pad_to_multiple(h_sample, multiple=16)
        rain_stack = pad_to_multiple(rain_stack, multiple=16)

        dem = torch.from_numpy(dem).unsqueeze(0)          # [1, H, W]
        rain = torch.from_numpy(rain_stack)               # [C, H, W]
        h_sample = torch.from_numpy(h_sample).unsqueeze(0)  # [1, H, W]

        x = torch.cat([dem, rain], dim=0)  # [C+1, H, W]

        return x, h_sample