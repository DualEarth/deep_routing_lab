import numpy as np
import random


def draw_ellipse_mask(mask, center, axes, angle):
    """Draw a filled ellipse onto a binary mask using NumPy only."""
    H, W = mask.shape
    cy, cx = center
    ay, ax = axes
    angle = np.deg2rad(angle)

    y, x = np.ogrid[:H, :W]
    x_shifted = x - cx
    y_shifted = y - cy

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x_rot = x_shifted * cos_a + y_shifted * sin_a
    y_rot = -x_shifted * sin_a + y_shifted * cos_a

    ellipse_mask = ((x_rot / ax) ** 2 + (y_rot / ay) ** 2) <= 1
    mask[ellipse_mask] = 1
    return mask


def generate_elliptical_cloud_mask(shape, max_coverage=0.3, max_ellipses=10):
    """
    Generate a binary cloud mask with elliptical shadows (NumPy-only).

    Args:
        shape (tuple): (H, W) of the image
        max_coverage (float): Maximum fraction of image area to be occluded (0 to 1)
        max_ellipses (int): Max number of ellipses to draw

    Returns:
        np.ndarray: Binary mask (1 = occluded, 0 = visible)
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)

    num_ellipses = random.randint(1, max_ellipses)
    target_area = random.uniform(0, max_coverage) * H * W
    current_area = 0

    for _ in range(num_ellipses):
        if current_area >= target_area:
            break

        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)
        ax = random.randint(W // 20, W // 5)
        ay = random.randint(H // 20, H // 5)
        angle = random.uniform(0, 360)

        previous_mask = mask.copy()
        mask = draw_ellipse_mask(mask, (cy, cx), (ay, ax), angle)
        current_area = np.count_nonzero(mask)

    return mask