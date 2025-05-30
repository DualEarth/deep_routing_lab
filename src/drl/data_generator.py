# src/drl/data_generator.py

import os
import numpy as np
import random
from tqdm import tqdm
from drl.utils import load_config, save_array_as_image, generate_elliptical_cloud_mask
from drl import DEMSimulator, RainfallSimulator, DiffusiveWaveRouter
from multiprocessing import Pool, cpu_count

def generate_one_sample(i, cfg, config_path, out_npz_dir, out_png_dir):
    dem_sim = DEMSimulator(config_path)
    rain_sim = RainfallSimulator(config_path)

    # Generate DEM
    dem = dem_sim.generate_dem()
    dem = dem_sim.trim_edges(dem)

    # Generate rainfall
    rain = rain_sim.generate(dem)
    T_rain, H, W = rain.shape

    # Routing
    if cfg["dataset"].get("use_momentum_routing", True):
        raise NotImplementedError("Shallow Water Router not yet implemented")
    else:
        router = DiffusiveWaveRouter(dem, config_path)

    h_sequence = router.run(rain)

    # Sample rain and output
    N = cfg["dataset"]["rain_snapshots"]
    S = cfg["dataset"]["snapshot_stride"]
    min_time = (N - 1) * S
    t_sample = random.randint(min_time, T_rain - 1)
    snapshot_indices = [t_sample - i * S for i in reversed(range(N))]
    rain_stack = np.stack([rain[t] for t in snapshot_indices], axis=0)
    h_sample = h_sequence[t_sample]

    # Cloud mask
    if cfg["dataset"].get("apply_cloud_mask", False):
        cloud_mask = generate_elliptical_cloud_mask((H, W), max_coverage=0.2)
        rain_stack[:, cloud_mask == 1] = 0.0

    # Save tensors
    npz_path = os.path.join(out_npz_dir, f"sample_{i:05d}.npz")
    np.savez_compressed(npz_path, dem=dem, rain=rain_stack, h_sample=h_sample)

    # Save images
    sample_png_dir = os.path.join(out_png_dir, f"sample_{i:05d}")
    os.makedirs(sample_png_dir, exist_ok=True)

    save_array_as_image(dem, os.path.join(sample_png_dir, "dem.png"), cmap="gray")
    for j, t in enumerate(snapshot_indices):
        save_array_as_image(rain[t], os.path.join(sample_png_dir, f"rain_{j:03d}.png"), cmap="Blues")
    save_array_as_image(h_sample, os.path.join(sample_png_dir, "h_sample.png"), cmap="Blues")

    return i  # return index for logging

def generate_training_dataset(config_path: str = './config.yaml', out_dir='dataset'):
    """
    Generate a dataset for training a surrogate pluvial flooding model.

    Each sample includes:
      - DEM [H, W]
      - Rain snapshots [N, H, W] (subsampled from storm)
      - Sample water depth [H, W] from overland routing

    Args:
        config_path (str): Path to YAML config file.
        out_dir (str): Directory to save output .npz files.
    """
    cfg = load_config(config_path)
    os.makedirs(out_dir, exist_ok=True)
    out_npz_dir = os.path.join(out_dir, "npz")
    out_png_dir = os.path.join(out_dir, "png")
    os.makedirs(out_npz_dir, exist_ok=True)
    os.makedirs(out_png_dir, exist_ok=True)

    num_samples = cfg["dataset"]["num_samples"]

    print(f"🛠 Using {cpu_count()} cores to generate {num_samples} samples...")

    args = [(i, cfg, config_path, out_npz_dir, out_png_dir) for i in range(num_samples)]

    with Pool() as pool:
        for i in tqdm(pool.starmap(generate_one_sample, args), total=num_samples, desc="Generating samples"):
            pass

    print(f"✅ Finished generating {num_samples} samples to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic flood training data")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to config file")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output directory")

    args = parser.parse_args()
    generate_training_dataset(config_path=args.config, out_dir=args.out_dir)