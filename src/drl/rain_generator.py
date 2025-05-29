# src/drl/rain_generator.py

import numpy as np
import random
from drl.utils import load_config

class RainfallSimulator:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        # self.intensity_range = (0.1, 1.0) 
        # self.speed_range = (1.0, 3.0)          # cells per timestep
        # self.sigma_range = (self.size/5, self.size/3)  # spatial std dev (in cells)
        self.rain_cfg = self.config["rainfall"]
        self.dem_cfg = self.config["dem"]
        self.size = self.dem_cfg["size"]
        self.intensity_range = tuple(self.rain_cfg["intensity_range"])
        self.speed_range = tuple(self.rain_cfg["speed_range"])
        self.sigma_range = (
            self.rain_cfg["sigma_fraction_range"][0] * self.size,
            self.rain_cfg["sigma_fraction_range"][1] * self.size,
        )

    def _generate_storm_params(self):
        intensity = random.uniform(*self.intensity_range)
        speed = random.uniform(*self.speed_range)
        sigma = random.uniform(*self.sigma_range)
        return intensity, speed, sigma

    def _get_direction_params(self, shape, direction):
        H, W = shape
        if direction == 'north':
            return (0, W // 2), (1, 0), H
        elif direction == 'south':
            return (H - 1, W // 2), (-1, 0), H
        elif direction == 'east':
            return (H // 2, W - 1), (0, -1), W
        elif direction == 'west':
            return (H // 2, 0), (0, 1), W
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def _generate_storm(self, dem, direction, intensity, speed, sigma):
        H, W = dem.shape
        (y0, x0), (dy, dx), travel_distance = self._get_direction_params((H, W), direction)
        duration = int(np.ceil(travel_distance / speed))
        rainfall = np.zeros((duration, H, W), dtype=np.float32)

        for t in range(duration):
            yt = int(round(y0 + t * speed * dy))
            xt = int(round(x0 + t * speed * dx))

            if not (0 <= yt < H and 0 <= xt < W):
                continue

            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            distance_sq = (yy - yt)**2 + (xx - xt)**2
            rainfall[t] = intensity * np.exp(-distance_sq / (2 * sigma**2))

        return rainfall

    def generate(self, dem: np.ndarray) -> np.ndarray:
        """Generate a rainfall field in a single random direction."""
        direction = random.choice(['north', 'south', 'east', 'west'])
        intensity, speed, sigma = self._generate_storm_params()
        return self._generate_storm(dem, direction, intensity, speed, sigma)

    def generate_all_directions(self, dem: np.ndarray) -> dict:
        """Generate rainfall fields for all directions using the same storm parameters."""
        intensity, speed, sigma = self._generate_storm_params()
        return {
            direction: self._generate_storm(dem, direction, intensity, speed, sigma)
            for direction in ['north', 'south', 'east', 'west']
        }