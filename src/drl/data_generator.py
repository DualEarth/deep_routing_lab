import os
import numpy as np
import random
from tqdm import tqdm
from drl.utils import load_config, save_array_as_image
from drl import DEMSimulator, RainfallSimulator, ShallowWaterRouter, DiffusiveWaveRouter

def generate_training_dataset(config_path: str = './config.config.yml', out_dir='dataset'):
    """
    Generate a dataset for training a surrogate pluvial flooding model.

    Each sample includes:
      - DEM [H, W]
      - Rain snapshots [N, H, W] (subsampled from storm)
      - Final water depth [H, W] from shallow water routing

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

    # Config values
    num_samples = cfg["dataset"]["num_samples"]
    n_snapshots = cfg["dataset"]["rain_snapshots"]
    stride = cfg["dataset"]["snapshot_stride"]
    use_momentum = cfg["dataset"].get("use_momentum_routing", True)

    # Initialize simulators
    dem_sim = DEMSimulator(config_path)
    rain_sim = RainfallSimulator(config_path)

    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Generate DEM
        dem = dem_sim.generate_dem()
        dem = dem_sim.trim_edges(dem)

        # Generate rainfall (random direction)
        rain = rain_sim.generate(dem)
        T_rain, H, W = rain.shape

        # Run routing
        if use_momentum:
            router = ShallowWaterRouter(dem, config_path)
        else:
            router = DiffusiveWaveRouter(dem, config_path)

        # Run routing to get full water‐depth sequence
        h_sequence = router.run(rain)

        # Total storm length and sampling parameters
        T_rain, H, W = rain.shape
        N = n_snapshots
        S = stride

        # Earliest we can sample so that we can include exactly N rain frames ending at t_sample
        min_time = (N - 1) * S

        # Pick a sample time between min_time and the last rain frame
        t_sample = random.randint(min_time, T_rain - 1)
        print(f"\n Sample of routing taken at {t_sample}")
        h_sample = h_sequence[t_sample]

        # Build N snapshot indices *ending* at t_sample, spaced by S
        # i=0 → t_sample - (N-1)*S ... i=N-1 → t_sample
        snapshot_indices = [t_sample - i * S for i in reversed(range(N))]
        print(f"Snapshots of precipitation leading up to sample: {snapshot_indices}")
        # e.g. N=5, S=2, t_sample=20 → [12,14,16,18,20]

        rain_stack = np.stack([rain[t] for t in snapshot_indices], axis=0)

        # Save raw tensors
        npz_path = os.path.join(out_npz_dir, f"sample_{i:05d}.npz")
        np.savez_compressed(npz_path,
                            dem=dem,
                            rain=rain_stack,
                            h_sample=h_sample)

        # Save images
        sample_png_dir = os.path.join(out_png_dir, f"sample_{i:05d}")
        os.makedirs(sample_png_dir, exist_ok=True)

        save_array_as_image(dem,
                            os.path.join(sample_png_dir, "dem.png"),
                            cmap="gray")
        for j, t in enumerate(snapshot_indices):
            save_array_as_image(rain[t],
                                os.path.join(sample_png_dir, f"rain_{j:03d}.png"),
                                cmap="Blues")
        save_array_as_image(h_sample,
                            os.path.join(sample_png_dir, "h_sample.png"),
                            cmap="Blues")

        router.reset()
    print(f"✅ Saved {num_samples} samples to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic flood training data")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to config file")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output directory")

    args = parser.parse_args()
    generate_training_dataset(config_path=args.config, out_dir=args.out_dir)