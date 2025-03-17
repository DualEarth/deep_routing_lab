import numpy as np
import scipy.ndimage

class DEMSimulator:
    def __init__(self, num_dems, size, hilliness_range=(1, 10), total_iterations=10, river_freq=2):
        """
        Initialize the DEM simulator with parameters for DEM generation, smoothing, and river carving.
        
        Args:
        num_dems (int): Number of DEMs to generate.
        size (int): Dimension of the DEMs (n x n).
        hilliness_range (tuple): Range of hilliness factors (min, max).
        total_iterations (int): Total number of iterations (smoothing + river carving).
        river_freq (int): Frequency of river carving steps (every 'river_freq' iterations).
        """
        self.num_dems = num_dems
        self.size = size
        self.hilliness_range = hilliness_range
        self.total_iterations = total_iterations
        self.river_freq = river_freq
        self.dems = []
 
    def generate_dem(self):
        """
        Generates a DEM with more randomized elevation variations while avoiding extreme sudden changes.
        """
        hilliness = np.random.uniform(*self.hilliness_range)
    
        # Step 1: Generate a base random elevation map with more natural variation
        dem = np.random.normal(50, hilliness * 5, (self.size, self.size))  # Start with random values around 50
    
        # Step 2: Apply Gaussian smoothing to remove abrupt changes
        dem = scipy.ndimage.gaussian_filter(dem, sigma=hilliness / 2)  

        # Step 3: Normalize elevation values to a reasonable range (0 - 100)
        dem = (dem - dem.min()) / (dem.max() - dem.min()) * 100  
    
        # Step 4: Perform iterative adjustments (optional, depending on visual results)
        for iteration in range(1, self.total_iterations + 1):
            if iteration % self.river_freq == 0:
                self.carve_river(dem)
            else:
                self.smooth_dem(dem)

        return dem

    def trim_edges(self, dem):
        """
        Trims the outer edges of the DEM to eliminate edge artifacts.
        """
        trimmed_size = self.size - 10  # Subtract 5 from each side
        return dem[5:-5, 5:-5]  # Skip the first and last 5 rows and columns

    def smooth_dem(self, dem):
        """
        Smooths the DEM by averaging selected cells with their neighbors.
        """
        num_cells_to_smooth = (self.size * self.size) // 2
        
        for _ in range(num_cells_to_smooth):
            i = np.random.randint(1, self.size-1)
            j = np.random.randint(1, self.size-1)
            neighborhood = dem[i-1:i+2, j-1:j+2]
            dem[i, j] = np.mean(neighborhood)

    def carve_river(self, dem):
        """
        Carves a river path from a randomly chosen edge point across the DEM according to a random slope.
        """
        # Start at a random edge of the DEM
        if np.random.rand() > 0.5:
            # Start from top/bottom edge
            i = np.random.choice([0, self.size-1])
            j = np.random.randint(self.size)
        else:
            # Start from left/right edge
            i = np.random.randint(self.size)
            j = np.random.choice([0, self.size-1])
        
        # Set the initial river cell to a low value
        dem[i, j] = np.min(dem)
        
        # Define the slope of the river
        vertical_step = np.random.choice([rnmb for rnmb in range(-10,10)])  # Move up (-1) or down (1)
        horizontal_step = np.random.choice([rnmb for rnmb in range(-10,10)])  # Move left (-1) or right (1)
        slope_ratio = np.random.randint(1, 4)  # Choose how many horizontal steps per vertical step

        steps = np.random.randint(0, self.size // 2)

        for _ in range(steps):
            # Move in the chosen direction according to the slope ratio
            for _ in range(slope_ratio):
                if 0 <= j + horizontal_step < self.size:
                    j += horizontal_step
                    dem[i, j] += np.random.uniform(0, 0.02) * dem[i, j]
                else:
                    break  # Stop if moving out of bounds

            if 0 <= i + vertical_step < self.size:
                i += vertical_step
                dem[i, j] += np.random.uniform(0, 0.02) * dem[i, j]
            else:
                break  # Stop if moving out of bounds

    def generate_all_dems(self):
        """
        Generate all DEMs as per the specified number.
        """
        for _ in range(self.num_dems):
            dem = self.generate_dem()
            trimmed_dem = self.trim_edges(dem)
            self.dems.append(trimmed_dem)

    def get_dems(self):
        """
        Returns the list of generated DEMs.
        """
        return self.dems
