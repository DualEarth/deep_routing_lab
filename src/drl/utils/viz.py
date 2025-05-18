# src/drl/utils/viz.py

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import LinearSegmentedColormap

bold_blue = LinearSegmentedColormap.from_list("bold_blue", ["#001f3f", "#0074D9", "#7FDBFF"])

def save_h(h_over_time, dem, out_dir='frames', vmin=0, vmax=None, cmap=bold_blue, alpha=0.6):
    """
    Save a sequence of water depth images as PNG files, overlaying h on top of the DEM.

    Args:
        h_over_time (list of np.ndarray): List of 2D arrays (water depths) over time.
        dem (np.ndarray): 2D array of elevations [L], same shape as h arrays.
        out_dir (str): Directory to save PNG files.
        vmin (float): Minimum color scale value for water depth.
        vmax (float): Maximum color scale value for water depth. If None, uses global max.
        cmap (str): Colormap for water.
        alpha (float): Transparency level for water layer (0 = invisible, 1 = opaque).
    """
    os.makedirs(out_dir, exist_ok=True)

    if vmax is None:
        vmax = max(np.max(h) for h in h_over_time)

    for t, h in enumerate(h_over_time):
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot DEM in grayscale
        ax.imshow(dem, cmap='gray', interpolation='none')

        # Overlay h with transparency
        im = ax.imshow(h, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation='none')

        # Colorbar for water depth only
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Water Depth [L]')

        ax.set_title(f"Timestep {t}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"frame_{t:04d}.png"))
        plt.close()

def make_gif_from_frames(out_dir='frames', gif_name='animation.gif', fps=5, cleanup=True):
    """
    Convert a sequence of PNG frames into a GIF and optionally delete the PNGs.

    Args:
        out_dir (str): Directory containing PNG frames.
        gif_name (str): Output GIF filename (saved in `out_dir`).
        fps (int): Frames per second for the GIF.
        cleanup (bool): Whether to delete PNG files after GIF creation.
    """
    frame_paths = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".png") and f.startswith("frame_")
    ])

    images = [imageio.v3.imread(f) for f in frame_paths]
    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, images, fps=fps)

    if cleanup:
        for f in frame_paths:
            os.remove(f)

    print(f"✅ GIF saved to {gif_path}")


# # Visualize rainfall at timestep 3
# for ts in range(100):
#     plt.figure(figsize=(4, 3))  # Width=4 inches, Height=3 inches
#     plt.imshow(rain[ts], cmap='Blues')
#     plt.colorbar(label='Rainfall Intensity (mm/hr)')
#     plt.title(f'Synthetic Rainfall at t={ts}')
#     plt.tight_layout()
#     plt.show()