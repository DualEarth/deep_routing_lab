import numpy as np
from tqdm import tqdm

class DiffusiveWaveRouter:
    def __init__(self, dem, dx=1.0, dt=0.1, manning_n=0.5):
        """
        Initialize the hydraulic router using diffusive wave approximation.

        Args:
            dem (np.ndarray): 2D elevation array [m]
            dx (float): grid spacing [m]
            dt (float): timestep [s]
            manning_n (float): Manning's roughness coefficient
        """
        self.dem = dem
        self.dx = dx
        self.dt = dt
        self.n = manning_n
        self.g = 9.81
        self.alpha = (1 / self.n) * (dx ** (2.0 / 3))  # Manning factor

        self.h = np.zeros_like(dem)  # initialize water depth
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