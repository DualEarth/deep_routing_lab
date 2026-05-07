# src/drl/utils/quality_control.py

import numpy as np


def check_mass_balance(sample_index, rain, h_final, total_boundary_outflow):
    """Compute and report a mass balance check for a single routed sample.

    Parameters
    ----------
    sample_index : int
        Index of the sample (used in the printed message).
    rain : np.ndarray
        Rainfall tensor of shape (T, H, W) in depth units [m per timestep].
    h_final : np.ndarray
        Water depth array at the end of routing, shape (H, W).
    total_boundary_outflow : float
        Cumulative volume that left through open boundaries during routing.
    """
    total_precip_input = float(np.sum(rain))
    total_remaining_volume = float(np.sum(h_final))
    total_boundary_outflow = float(total_boundary_outflow)

    mass_balance_residual = (
        total_precip_input - total_boundary_outflow
    ) - total_remaining_volume
    pct_residual = mass_balance_residual / (total_precip_input + 1e-12) * 100.0

    if abs(pct_residual) < 1.0:
        print(f"Sample {sample_index:05d} mass balance check passed.")
    else:
        print(
            f"Sample {sample_index:05d} WARNING: Mass balance error is"
            f" {pct_residual:.4f}%."
        )
