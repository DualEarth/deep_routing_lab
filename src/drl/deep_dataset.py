import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from drl.utils.tensor_ops import pad_to_multiple


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
    def __init__(self, root_dir, num_rain_frames=5, apply_cloud_mask=False, cloud_mask_fn=None):
        self.root_dir = root_dir
        self.samples = sorted([d for d in os.listdir(root_dir) if d.startswith("sample_")])
        self.num_rain_frames = num_rain_frames
        self.apply_cloud_mask = apply_cloud_mask
        self.cloud_mask_fn = cloud_mask_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        sample_dir = os.path.join(self.root_dir, sample_name)

        dem = load_image(os.path.join(sample_dir, "dem.png"))
        assert dem.ndim == 2, f"DEM shape wrong: {dem.shape}"
        h_final = load_image(os.path.join(sample_dir, "h_final.png"))
        rain_stack = stack_rain_images(sample_dir, self.num_rain_frames)

        if self.apply_cloud_mask and self.cloud_mask_fn is not None:
            mask = self.cloud_mask_fn(h_final.shape)
            h_final = np.where(mask == 1, 0.0, h_final)

        assert h_final.ndim == 2, f"h_final shape wrong: {h_final.shape}"
        assert rain_stack.ndim == 3, f"Rain stack shape wrong: {rain_stack.shape}"

        # load dem, h_final, rain_stack as np arrays...
        dem = pad_to_multiple(dem, multiple=16)
        h_final = pad_to_multiple(h_final, multiple=16)
        rain_stack = pad_to_multiple(rain_stack, multiple=16)

        dem = torch.from_numpy(dem).unsqueeze(0)          # [1, H, W]
        rain = torch.from_numpy(rain_stack)               # [C, H, W]
        h_final = torch.from_numpy(h_final).unsqueeze(0)  # [1, H, W]

        x = torch.cat([dem, rain], dim=0)  # [C+1, H, W]

        return x, h_final