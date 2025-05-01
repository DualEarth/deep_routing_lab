import numpy as np
import random

class RainfallSimulator:
    def __init__(self):
        # Hardcoded simulation parameters
        self.duration = 24                     # number of timesteps
        self.intensity_range = (10.0, 50.0)    # mm/hr
        self.speed_range = (1.0, 3.0)          # cells per timestep
        self.sigma_range = (3.0, 8.0)          # spatial std dev (in cells)

    def _select_origin(self, shape):
        """Choose a storm entry direction and starting point centered on the domain edge."""
        direction = random.choice(['north', 'south', 'east', 'west'])
        H, W = shape
        if direction == 'north':
            pos = (0, W // 2)
            move_vector = (1, 0)
        elif direction == 'south':
            pos = (H - 1, W // 2)
            move_vector = (-1, 0)
        elif direction == 'east':
            pos = (H // 2, W - 1)
            move_vector = (0, -1)
        elif direction == 'west':
            pos = (H // 2, 0)
            move_vector = (0, 1)
        return direction, pos, move_vector

    def _generate_storm_params(self):
        intensity = random.uniform(*self.intensity_range)
        speed = random.uniform(*self.speed_range)
        sigma = random.uniform(*self.sigma_range)
        return intensity, speed, sigma

    def generate(self, dem: np.ndarray) -> np.ndarray:
        """Generate rainfall as a 3D array (time, height, width)."""
        H, W = dem.shape
        rainfall = np.zeros((self.duration, H, W), dtype=np.float32)

        direction, (y0, x0), (dy, dx) = self._select_origin((H, W))
        intensity, speed, sigma = self._generate_storm_params()

        for t in range(self.duration):
            yt = int(round(y0 + t * speed * dy))
            xt = int(round(x0 + t * speed * dx))

            if not (0 <= yt < H and 0 <= xt < W):
                break  # storm has exited the domain

            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            distance_sq = (yy - yt)**2 + (xx - xt)**2
            rainfall[t] = intensity * np.exp(-distance_sq / (2 * sigma**2))

        return rainfall