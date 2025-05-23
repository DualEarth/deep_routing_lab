# src/drl/utils/viz.py

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

def plot_comparison(y_true, y_pred, index=0, save_dir="validation_plots"):

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"comparison_{index:03d}.png")

    # If shape is [1, H, W] or [C, H, W], squeeze/convert to [H, W]
    if y_true.ndim == 3:
        y_true = y_true[0]
    if y_pred.ndim == 3:
        y_pred = y_pred[0]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(y_true, cmap='Blues')
    plt.title(f'True h_sample #{index}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(y_pred, cmap='Blues')
    plt.title(f'Predicted h_sample #{index}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_array_as_image(arr, path, cmap='gray', vmin=None, vmax=None):
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)

def save_h(h_over_time, dem, out_dir='outputs/routing', vmin=0, vmax=None, cmap="Blues", alpha=0.6, n_contours=20):
    """
    Save a sequence of water depth images as PNG files, overlaying h on top of DEM contours.

    Args:
        h_over_time (list of np.ndarray): List of 2D arrays (water depths) over time.
        dem (np.ndarray): 2D array of elevations [L], same shape as h arrays.
        out_dir (str): Directory to save PNG files.
        vmin (float): Minimum color scale value for water depth.
        vmax (float): Maximum color scale value for water depth. If None, uses global max.
        cmap (str): Colormap for water.
        alpha (float): Transparency level for water layer (0 = invisible, 1 = opaque).
        n_contours (int): Number of topographic contour lines to draw.
    """
    os.makedirs(out_dir, exist_ok=True)

    if vmax is None:
        vmax = max(np.max(h) for h in h_over_time)

    for t, h in enumerate(h_over_time):
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot DEM contours
        contour_levels = np.linspace(np.min(dem), np.max(dem), n_contours)
        ax.contour(dem, levels=contour_levels, colors='black', linewidths=0.5, alpha=0.6)

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

def make_routing_gif(out_dir='outputs/routing', gif_name='animation.gif', fps=5, cleanup=True):
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


def make_rainfall_gif(rain, dem, out_dir='outputs/rain', gif_name='rainfall.gif', fps=5, cleanup=True, n_contours=20):
    """
    Create a GIF showing the progression of rainfall over a DEM topographic contour map.

    Args:
        rain (np.ndarray): 3D rainfall array [time, H, W]
        dem (np.ndarray): 2D elevation array [H, W]
        out_dir (str): Directory to save PNG frames.
        gif_name (str): Output GIF filename (saved in `out_dir`).
        fps (int): Frames per second for GIF.
        cleanup (bool): Whether to delete PNG files after GIF creation.
        n_contours (int): Number of topographic contour lines.
    """
    os.makedirs(out_dir, exist_ok=True)

    T, H, W = rain.shape
    vmax = np.max(rain)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(6, 5))

        # DEM topographic contours
        contour_levels = np.linspace(np.min(dem), np.max(dem), n_contours)
        ax.contour(dem, levels=contour_levels, colors='black', linewidths=0.5, alpha=0.6)

        # Rainfall layer
        im = ax.imshow(rain[t], cmap='Blues', vmin=0, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Rainfall Intensity [L/T]')

        ax.set_title(f"Rainfall - Timestep {t}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rain_{t:04d}.png"))
        plt.close()

    # Assemble GIF
    frame_paths = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".png") and f.startswith("rain_")
    ])
    images = [imageio.v3.imread(f) for f in frame_paths]
    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, images, fps=fps)

    if cleanup:
        for f in frame_paths:
            os.remove(f)

    print(f"✅ Rainfall GIF saved to {gif_path}")

def make_quad_rainfall_gif(rain_dict, dem, out_dir='outputs/rain', gif_name='rainfall_quadrants.gif',
                                fps=5, cleanup=True, n_contours=20):
    """
    Create a GIF showing all four rainfall directions (north, south, east, west) as quadrants over a DEM.

    Args:
        rain_dict (dict): Dictionary of {direction: rainfall [T, H, W]}
        dem (np.ndarray): 2D elevation array [H, W]
        out_dir (str): Directory to save PNG frames.
        gif_name (str): Output GIF filename.
        fps (int): Frames per second for GIF.
        cleanup (bool): Whether to delete intermediate PNGs.
        n_contours (int): Number of contour lines for DEM.
    """

    os.makedirs(out_dir, exist_ok=True)

    directions = ["north", "south", "east", "west"]
    T = max(rain_dict[d].shape[0] for d in directions)
    H, W = dem.shape
    vmax = max(np.max(rain_dict[d]) for d in directions)

    for t in range(T):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        ax_map = {
            "north": axes[0, 0],
            "south": axes[1, 0],
            "east":  axes[0, 1],
            "west":  axes[1, 1]
        }

        for dir_name, ax in ax_map.items():
            rainfall = rain_dict[dir_name]
            if t < rainfall.shape[0]:
                frame = rainfall[t]
            else:
                frame = np.zeros_like(dem)

            # Draw DEM contours
            levels = np.linspace(np.min(dem), np.max(dem), n_contours)
            ax.contour(dem, levels=levels, colors='black', linewidths=0.5, alpha=0.6)

            im = ax.imshow(frame, cmap='Blues', vmin=0, vmax=vmax)
            ax.set_title(f"{dir_name.capitalize()} (t={t})")
            ax.axis('off')

        # Add colorbar outside the plot
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        fig.colorbar(im, cax=cbar_ax, label='Rainfall Intensity [L/T]')

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        frame_path = os.path.join(out_dir, f"rainquad_{t:04d}.png")
        plt.savefig(frame_path)
        plt.close()

    # Make GIF
    frame_paths = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("rainquad_") and f.endswith(".png")
    ])
    images = [imageio.v3.imread(f) for f in frame_paths]
    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, images, fps=fps)

    if cleanup:
        for f in frame_paths:
            os.remove(f)

    print(f"✅ Quadrant rainfall GIF saved to {gif_path}")

def save_h_quads(h_dict, dem, out_dir='outputs/routing', vmin=0, vmax=None, alpha=0.6, n_contours=20, cmap="Blues"):
    """
    Save frames showing water depth evolution from four directions (N, S, E, W) in quad layout.

    Args:
        h_dict (dict): {direction: list of 2D h arrays}
        dem (np.ndarray): Elevation grid [H, W]
        out_dir (str): Where to save PNG frames
        vmin (float): Min value for h colormap
        vmax (float): Max value for h colormap
        alpha (float): Alpha transparency of h layer
        n_contours (int): Number of DEM contours
    """

    os.makedirs(out_dir, exist_ok=True)

    directions = ["north", "south", "east", "west"]
    T = max(len(h_dict[d]) for d in directions)
    H, W = dem.shape
    if vmax is None:
        vmax = max(np.max(h) for direction in h_dict for h in h_dict[direction])

    for t in range(T):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        ax_map = {
            "north": axes[0, 0],
            "south": axes[1, 0],
            "east":  axes[0, 1],
            "west":  axes[1, 1]
        }

        for dir_name, ax in ax_map.items():
            h_list = h_dict[dir_name]
            h = h_list[t] if t < len(h_list) else np.zeros_like(dem)

            # DEM contours
            levels = np.linspace(np.min(dem), np.max(dem), n_contours)
            ax.contour(dem, levels=levels, colors='black', linewidths=0.5, alpha=0.6)

            # Water depth
            im = ax.imshow(h, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
            ax.set_title(f"{dir_name.capitalize()} (t={t})")
            ax.axis('off')

        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        fig.colorbar(im, cax=cbar_ax, label='Water Depth [L]')

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(out_dir, f"routingquad_{t:04d}.png"))
        plt.close()

    print(f"✅ Quadrant routing images saved to {out_dir}")
    
def make_quad_routing_gif(out_dir='outputs/routing', gif_name='routing_quads.gif', fps=5, cleanup=True):
    """
    Create a GIF from quadrant routing images saved in `save_h_scenarios`.

    Args:
        out_dir (str): Directory containing routing frames
        gif_name (str): Name of output GIF
        fps (int): Frames per second
        cleanup (bool): Whether to delete PNGs after making GIF
    """

    frame_paths = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith('.png') and f.startswith('routingquad_')
    ])
    images = [imageio.v3.imread(f) for f in frame_paths]
    gif_path = os.path.join(out_dir, gif_name)
    imageio.mimsave(gif_path, images, fps=fps)

    if cleanup:
        for f in frame_paths:
            os.remove(f)

    print(f"✅ Quadrant routing GIF saved to {gif_path}")