# src/drl/rain_generator.py

import numpy as np
import random
from math import erf
from drl.utils import load_config


DIRECTIONS_8 = [
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
]

class RainfallSimulator:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.rain_cfg = self.config["rainfall"]
        self.dem_cfg = self.config["dem"]
        self.size = self.dem_cfg["size"]
        self.intensity_range = tuple(self.rain_cfg["intensity_range"])
        self.speed_range = tuple(self.rain_cfg["speed_range"])
        self.sigma_range = (
            self.rain_cfg["sigma_fraction_range"][0] * self.size,
            self.rain_cfg["sigma_fraction_range"][1] * self.size,
        )
        self.storm_shape = str(self.rain_cfg.get("storm_shape", "blob")).lower()
        if self.storm_shape not in ("blob", "flat"):
            raise ValueError(
                f"Invalid rainfall.storm_shape '{self.storm_shape}'. Expected 'blob' or 'flat'."
            )
        self.edge_smoothing_fraction = float(self.rain_cfg.get("edge_smoothing_fraction", 0.25))
        self.min_edge_smoothing_cells = float(self.rain_cfg.get("min_edge_smoothing_cells", 0.5))

    def _erf_array(self, values):
        return np.vectorize(erf, otypes=[np.float64])(values)

    def _leading_trailing_taper(self, proj_parallel, front_thickness, sigma):
        """Gaussian-smoothed box profile along travel direction."""
        edge_sigma = max(sigma * self.edge_smoothing_fraction, self.min_edge_smoothing_cells)
        if edge_sigma <= 0:
            return ((proj_parallel >= 0.0) & (proj_parallel <= front_thickness)).astype(np.float32)

        norm = np.sqrt(2.0) * edge_sigma
        leading = self._erf_array(proj_parallel / norm)
        trailing = self._erf_array((proj_parallel - front_thickness) / norm)
        taper = 0.5 * (leading - trailing)
        return np.clip(taper, 0.0, 1.0).astype(np.float32)

    def _generate_storm_params(self):
        intensity = random.uniform(*self.intensity_range)
        speed = random.uniform(*self.speed_range)
        sigma = random.uniform(*self.sigma_range)
        return intensity, speed, sigma

    def _get_direction_params(self, shape, direction):
        H, W = shape
        diag = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))
        params = {
            "north": ((0, W // 2), (1, 0), H),
            "south": ((H - 1, W // 2), (-1, 0), H),
            "east": ((H // 2, W - 1), (0, -1), W),
            "west": ((H // 2, 0), (0, 1), W),
            "northeast": ((0, 0), (1, 1), diag),
            "southeast": ((H - 1, 0), (-1, 1), diag),
            "southwest": ((H - 1, W - 1), (-1, -1), diag),
            "northwest": ((0, W - 1), (1, -1), diag),
        }
        if direction not in params:
            raise ValueError(f"Invalid direction: {direction}")
        start, (dy, dx), dist = params[direction]
        mag = np.sqrt(dy ** 2 + dx ** 2)
        return start, (dy / mag, dx / mag), dist

    def _generate_storm(self, dem, direction, intensity, speed, sigma):
        H, W = dem.shape
        (y0, x0), (dy, dx), travel_distance = self._get_direction_params((H, W), direction)
        duration = int(np.ceil(travel_distance / speed))
        rainfall = np.zeros((duration, H, W), dtype=np.float32)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        if self.storm_shape == "flat":
            if direction in ["north", "south"]:
                front_span = W
            elif direction in ["east", "west"]:
                front_span = H
            else:
                front_span = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))
            front_thickness = 2 * sigma
            half_span = front_span / 2.0

        for t in range(duration):
            yt = y0 + t * speed * dy
            xt = x0 + t * speed * dx

            if self.storm_shape == "blob":
                distance_sq = (yy - yt) ** 2 + (xx - xt) ** 2
                rainfall[t] = intensity * np.exp(-distance_sq / (2 * sigma ** 2))
            else:
                proj_parallel = (yy - yt) * dy + (xx - xt) * dx
                proj_perp = (yy - yt) * (-dx) + (xx - xt) * dy

                side_mask = (np.abs(proj_perp) <= half_span).astype(np.float32)
                edge_taper = self._leading_trailing_taper(proj_parallel, front_thickness, sigma)
                rainfall[t] = intensity * side_mask * edge_taper

        return rainfall

    def generate(self, dem: np.ndarray) -> np.ndarray:
        """Generate a rainfall field in a single random direction."""
        direction = random.choice(DIRECTIONS_8)
        intensity, speed, sigma = self._generate_storm_params()
        return self._generate_storm(dem, direction, intensity, speed, sigma)

    def generate_all_directions(self, dem: np.ndarray) -> dict:
        """Generate rainfall fields for all 8 directions using the same storm parameters."""
        intensity, speed, sigma = self._generate_storm_params()
        return {
            direction: self._generate_storm(dem, direction, intensity, speed, sigma)
            for direction in DIRECTIONS_8
        }