import numpy as np
import random
from drl.utils import load_config

class RainfallSimulator:
    def __init__(self, config_path):
        # Hardcoded simulation parameters
        self.config = load_config(config_path)
        self.size = self.config["size"]
        self.intensity_range = (0.1, 1.0) 
        self.speed_range = (1.0, 3.0)          # cells per timestep
        self.sigma_range = (self.size/5, self.size/3)          # spatial std dev (in cells)

    def _generate_storm_params(self):
        intensity = random.uniform(*self.intensity_range)
        speed = random.uniform(*self.speed_range)
        sigma = random.uniform(*self.sigma_range)
        return intensity, speed, sigma
    
    def generate(self, dem: np.ndarray) -> dict:
        """
        Generate rainfall fields for all four directions (N, S, E, W) using same storm parameters.

        Args:
            dem (np.ndarray): Elevation grid [H, W]

        Returns:
            dict: Direction → rainfall 3D array [T, H, W]
        """
        H, W = dem.shape

        intensity, speed, sigma = self._generate_storm_params()

        params = {
            'north': ((0, W // 2), (1, 0), H),
            'south': ((H - 1, W // 2), (-1, 0), H),
            'east':  ((H // 2, W - 1), (0, -1), W),
            'west':  ((H // 2, 0), (0, 1), W)
        }

        rain_by_direction = {}
        for direction, ((y0, x0), (dy, dx), travel_distance) in params.items():
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

            rain_by_direction[direction] = rainfall

        return rain_by_direction