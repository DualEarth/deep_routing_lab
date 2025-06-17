import torch
import numpy as np


def center_crop_to_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Center-crop `source` to match spatial dimensions of `target`.

    Args:
        source (torch.Tensor): Tensor with shape (B, C, Hs, Ws)
        target (torch.Tensor): Tensor with shape (B, C, Ht, Wt)

    Returns:
        torch.Tensor: Cropped source tensor of shape (B, C, Ht, Wt)
    """
    _, _, Hs, Ws = source.shape
    _, _, Ht, Wt = target.shape

    start_y = (Hs - Ht) // 2
    start_x = (Ws - Wt) // 2

    return source[:, :, start_y:start_y + Ht, start_x:start_x + Wt]


def pad_to_multiple(arr: np.ndarray, multiple: int = 16) -> np.ndarray:
    """
    Pad a 2D or 3D array symmetrically so that its H and W become multiples of `multiple`.
    Pads with zeros.

    Args:
        arr (np.ndarray): Array with shape [H, W] or [C, H, W]
        multiple (int): The target multiple for dimensions.

    Returns:
        np.ndarray: Padded array.
    """
    H, W = arr.shape[-2], arr.shape[-1]
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if arr.ndim == 2:
        return np.pad(arr, ((top, bottom), (left, right)), mode='constant')
    elif arr.ndim == 3:
        # e.g. rain stack [C, H, W]
        return np.pad(arr, ((0, 0), (top, bottom), (left, right)), mode='constant')
    else:
        raise ValueError(f"Cannot pad array with shape {arr.shape}")


def pad_or_crop_array_to_match(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Symmetrically pad or crop a 2D array to match a target shape.

    Args:
        arr (np.ndarray): Input 2D array with shape (H, W).
        target_shape (tuple): Desired output shape (Ht, Wt).

    Returns:
        np.ndarray: Array of shape (Ht, Wt), padded or cropped as needed.
    """
    H, W = arr.shape
    Ht, Wt = target_shape

    # if array larger, center-crop
    if H > Ht or W > Wt:
        start_y = max((H - Ht) // 2, 0)
        start_x = max((W - Wt) // 2, 0)
        cropped = arr[start_y:start_y + Ht, start_x:start_x + Wt]
    else:
        cropped = arr

    # pad if smaller
    pad_h = max(Ht - cropped.shape[0], 0)
    pad_w = max(Wt - cropped.shape[1], 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(cropped, ((top, bottom), (left, right)), mode='constant')
