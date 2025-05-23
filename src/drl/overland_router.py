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
        R = rain_input  # [m/s] rainfall rate

        # --- 1) Estimate max transport & wave speed for CFL ---
        # safe water depth
        h_safe = np.maximum(self.h, 1e-4)
        # water surface
        zsurf = self.dem + h_safe
        # surface slopes
        dzdx = (zsurf[:,1:] - zsurf[:,:-1]) / self.dx
        dzdy = (zsurf[1:,:] - zsurf[:-1,:]) / self.dx
        # clip slopes
        dzdx = np.clip(dzdx, -10, 10)
        dzdy = np.clip(dzdy, -10, 10)
        # face depths
        hfx = np.minimum(h_safe[:,1:], h_safe[:,:-1])
        hfy = np.minimum(h_safe[1:,:], h_safe[:-1,:])
        # approximate velocities u = q/h, q = -h^(5/3) * slope / n
        qx = -(1/self.n) * (hfx**(5/3)) * dzdx
        qy = -(1/self.n) * (hfy**(5/3)) * dzdy
        ux = np.abs(qx) / hfx
        uy = np.abs(qy) / hfy
        # shallow‐water wave speed
        c = np.sqrt(self.g * h_safe)
        # maximum signal speed
        max_signal = np.max(ux) + np.max(uy) + np.max(c)
        # CFL limit: dt_sub <= dx / max_signal * safety
        safety = 0.5
        dt_max = safety * self.dx / (max_signal + 1e-6)
        # number of substeps
        n_sub = max(1, int(np.ceil(self.dt / dt_max)))
        dt_sub = self.dt / n_sub

        # --- 2) Run n_sub explicit updates at dt_sub each ---
        for _ in range(n_sub):
            # recompute water surface
            z = self.dem + self.h
            H, W = self.h.shape
            # zero‐flux arrays
            qx = np.zeros((H, W+1))
            qy = np.zeros((H+1, W))

            # compute slopes & fluxes exactly as before
            dzdx = (z[:,1:] - z[:,:-1]) / self.dx
            dzdy = (z[1:,:] - z[:-1,:]) / self.dx
            dzdx = np.clip(dzdx, -10, 10)
            dzdy = np.clip(dzdy, -10, 10)

            hfx = np.minimum(self.h[:,1:], self.h[:,:-1])
            hfy = np.minimum(self.h[1:,:], self.h[:-1,:])
            hfx = np.clip(hfx, 0, 10)
            hfy = np.clip(hfy, 0, 10)

            qface_x = -(1/self.n) * (hfx**(5/3)) * dzdx
            qface_y = -(1/self.n) * (hfy**(5/3)) * dzdy
            qx[:,1:-1] = np.where(dzdx < 0, qface_x, 0)
            qy[1:-1,:] = np.where(dzdy < 0, qface_y, 0)

            # divergence
            dqx = (qx[:,1:] - qx[:,:-1]) / self.dx
            dqy = (qy[1:,:] - qy[:-1,:]) / self.dx
            div_q = dqx + dqy

            # update depth
            dh_dt = R - div_q
            self.h += dt_sub * dh_dt
            self.h = np.nan_to_num(self.h, nan=0.0, posinf=0.0, neginf=0.0)
            self.h = np.maximum(self.h, 0.0)

        # --- 3) record only once per original dt ---
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

        for t in tqdm(range(T_total), desc="Diffusive Routing"):
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
        Momentum-aware shallow water routing model with adaptive sub-stepping for stability.

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
        R = rain_input  # rainfall rate [m/s]
        # --- 1) CFL check for adaptive sub-stepping ---
        # ensure nonzero depth
        h_safe = np.maximum(self.h, 1e-4)
        # wave speed
        c = np.sqrt(self.g * h_safe)
        # signal speed = |u| + c
        signal = np.abs(self.ux) + np.abs(self.uy) + c
        max_signal = np.max(signal)
        safety = 0.5
        dt_max = safety * self.dx / (max_signal + 1e-6)
        n_sub = max(1, int(np.ceil(self.dt / dt_max)))
        dt_sub = self.dt / n_sub

        # --- 2) Sub-stepping loop ---
        if n_sub > 1000:
            n_sub = 1000
        for _ in range(n_sub):
            z = self.dem
            h = self.h
            ux = self.ux
            uy = self.uy

            # Stability: clip
            h_safe = np.maximum(h, 1e-4)
            ux = np.clip(ux, -10, 10)
            uy = np.clip(uy, -10, 10)

            # Surface slope
            eta = z + h
            dzdx = (np.roll(eta, -1, axis=1) - np.roll(eta, 1, axis=1)) / (2 * self.dx)
            dzdy = (np.roll(eta, -1, axis=0) - np.roll(eta, 1, axis=0)) / (2 * self.dx)

            # Friction
            vel = np.sqrt(ux**2 + uy**2)
            Sf_x = self.n**2 * ux * vel / (h_safe**(4/3))
            Sf_y = self.n**2 * uy * vel / (h_safe**(4/3))

            # Momentum update
            ux_new = ux - self.g * dt_sub * dzdx - dt_sub * Sf_x
            uy_new = uy - self.g * dt_sub * dzdy - dt_sub * Sf_y

            # Continuity (flux divergence)
            dhdx = (np.roll(ux_new * h_safe, -1, axis=1) - np.roll(ux_new * h_safe, 1, axis=1)) / (2 * self.dx)
            dhdy = (np.roll(uy_new * h_safe, -1, axis=0) - np.roll(uy_new * h_safe, 1, axis=0)) / (2 * self.dx)
            dh_dt = R - (dhdx + dhdy)

            # Update
            h = h + dt_sub * dh_dt
            h = np.clip(h, 0, None)

            # Save for next substep
            self.h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            self.ux = np.nan_to_num(ux_new, nan=0.0, posinf=0.0, neginf=0.0)
            self.uy = np.nan_to_num(uy_new, nan=0.0, posinf=0.0, neginf=0.0)

        # record only once per original dt
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
        """Reset water depth and history."""
        self.h = np.zeros_like(self.dem)
        self.ux = np.zeros_like(self.dem)
        self.uy = np.zeros_like(self.dem)
        self.h_over_time = []