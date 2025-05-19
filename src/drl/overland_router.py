import numpy as np
from tqdm import tqdm
from drl.utils import load_config

class DiffusiveWaveRouter:
    def __init__(self, dem, config_path):
        """
        Initialize the hydraulic router using parameters from config.

        Args:
            dem (np.ndarray): 2D elevation array [m]
            config_path (str, optional): Path to YAML config file. If provided, overrides dx, dt, and n.
        """
        self.dem = dem

        cfg = load_config(config_path)
        routing_cfg = cfg["routing"]
        self.dx = routing_cfg["dx"]
        self.dt = routing_cfg["dt"]
        self.n = routing_cfg["manning_n"]

        self.g = 9.81
        self.alpha = (1 / self.n) * (self.dx ** (2.0 / 3))  # Manning factor

        self.h = np.zeros_like(dem)
        self.h_over_time = []

    def step(self, rain_input):
        R = rain_input
        z = self.dem + self.h

        H, W = self.h.shape
        qx = np.zeros((H, W+1))
        qy = np.zeros((H+1, W))

        # Flow left-right (east-west)
        dzdx = (z[:, 1:] - z[:, :-1]) / self.dx
        dzdx = np.clip(dzdx, -10, 10)
        h_face_x = np.minimum(self.h[:, 1:], self.h[:, :-1])
        h_face_x = np.clip(h_face_x, 0, 10)
        q_face_x = -(1 / self.n) * h_face_x ** (5/3) * dzdx
        qx[:, 1:-1] = np.where(dzdx < 0, q_face_x, 0)

        # Flow up-down (north-south)
        dzdy = (z[1:, :] - z[:-1, :]) / self.dx
        dzdy = np.clip(dzdy, -10, 10)
        h_face_y = np.minimum(self.h[1:, :], self.h[:-1, :])
        h_face_y = np.clip(h_face_y, 0, 10)
        q_face_y = -(1 / self.n) * h_face_y ** (5/3) * dzdy
        qy[1:-1, :] = np.where(dzdy < 0, q_face_y, 0)

        # Divergence (out - in)
        dqx = (qx[:, 1:] - qx[:, :-1]) / self.dx
        dqy = (qy[1:, :] - qy[:-1, :]) / self.dx
        div_q = dqx + dqy

        # Update depth
        dh_dt = R - div_q

        self.h += self.dt * dh_dt
        self.h = np.nan_to_num(self.h, nan=0.0, posinf=0.0, neginf=0.0)

        self.h = np.maximum(self.h, 0)
        self.h_over_time.append(self.h.copy())

        if np.any(np.isnan(self.h)) or np.any(np.isinf(self.h)):
            print("⚠️ NaNs or Infs detected in h at timestep.")

    def run(self, rainfall_3d):
        """
        Run the routing model for a full rainfall time series, extended to allow draining.

        Args:
            rainfall_3d (np.ndarray): Rainfall [time, H, W]

        Returns:
            List[np.ndarray]: List of water depth fields over time (saved every `save_every` steps)
        """
        self.h_over_time = []

        T_rain, H, W = rainfall_3d.shape
        T_total = T_rain * 2
        T_extra = T_total - T_rain

        rain_extended = np.concatenate([
            rainfall_3d,
            np.zeros((T_extra, H, W), dtype=rainfall_3d.dtype)
        ])

        for t in tqdm(range(T_total), desc="Routing"):
            self.step(rain_extended[t])
            self.h_over_time.append(self.h.copy())

        return self.h_over_time

    def reset(self):
        """Reset water depth and history."""
        self.h = np.zeros_like(self.dem)
        self.h_over_time = []



class ShallowWaterRouter:
    def __init__(self, dem, config_path):
        """
        Momentum-aware shallow water routing model.

        Args:
            dem (np.ndarray): 2D elevation array [m]
            config_path (str): YAML config file with routing parameters.
        """
        self.dem = dem

        cfg = load_config(config_path)
        routing_cfg = cfg["routing"]
        self.dx = routing_cfg["dx"]
        self.dt = routing_cfg["dt"]
        self.n = routing_cfg["manning_n"]

        self.g = 9.81
        self.h = np.zeros_like(dem)   # water depth [m]
        self.ux = np.zeros_like(dem)  # velocity in x
        self.uy = np.zeros_like(dem)  # velocity in y
        self.h_over_time = []

    def step(self, rain_input):
        R = rain_input
        z = self.dem
        h = self.h
        ux = self.ux
        uy = self.uy

        # --- Stability: clip velocities and ensure safe depth ---
        h_safe = np.maximum(h, 1e-4)
        ux = np.clip(ux, -10, 10)
        uy = np.clip(uy, -10, 10)

        # Compute slope of water surface
        eta = z + h
        dzdx = (np.roll(eta, -1, axis=1) - np.roll(eta, 1, axis=1)) / (2 * self.dx)
        dzdy = (np.roll(eta, -1, axis=0) - np.roll(eta, 1, axis=0)) / (2 * self.dx)

        # Friction terms
        velocity_mag = np.sqrt(ux**2 + uy**2)
        Sf_x = self.n**2 * ux * velocity_mag / (h_safe**(4/3))
        Sf_y = self.n**2 * uy * velocity_mag / (h_safe**(4/3))

        # Momentum update
        ux_new = ux - self.g * self.dt * dzdx - self.dt * Sf_x
        uy_new = uy - self.g * self.dt * dzdy - self.dt * Sf_y

        # Flux divergence
        dhdx = (np.roll(ux_new * h_safe, -1, axis=1) - np.roll(ux_new * h_safe, 1, axis=1)) / (2 * self.dx)
        dhdy = (np.roll(uy_new * h_safe, -1, axis=0) - np.roll(uy_new * h_safe, 1, axis=0)) / (2 * self.dx)
        dh_dt = R - (dhdx + dhdy)

        # Water depth update
        h_new = h + self.dt * dh_dt
        h_new = np.clip(h_new, 0, None)

        # Final sanity check
        self.h = np.nan_to_num(h_new, nan=0.0, posinf=0.0, neginf=0.0)
        self.ux = np.nan_to_num(ux_new, nan=0.0, posinf=0.0, neginf=0.0)
        self.uy = np.nan_to_num(uy_new, nan=0.0, posinf=0.0, neginf=0.0)

        self.h_over_time.append(self.h.copy())

    def run(self, rainfall_3d):
        """
        Run the model for the full storm + drainage.

        Args:
            rainfall_3d (np.ndarray): [time, H, W]
        Returns:
            List[np.ndarray]: water depth snapshots over time
        """
        self.h_over_time = []

        T_rain, H, W = rainfall_3d.shape
        T_total = 2 * T_rain

        rain_extended = np.concatenate([
            rainfall_3d,
            np.zeros((T_total - T_rain, H, W), dtype=rainfall_3d.dtype)
        ])

        for t in tqdm(range(T_total), desc="Momentum Routing"):
            self.step(rain_extended[t])

        return self.h_over_time

    def reset(self):
        self.h = np.zeros_like(self.dem)
        self.ux = np.zeros_like(self.dem)
        self.uy = np.zeros_like(self.dem)
        self.h_over_time = []