import numpy as np
from config import num_images, image_size, batch_size, num_images_train
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_synthetic_data():
    # Initialize arrays to store synthetic data
    X1 = np.zeros((num_images, image_size, image_size), dtype=np.float32)
    X2 = np.zeros((num_images, image_size, image_size), dtype=np.float32)
    y = np.zeros((num_images, image_size, image_size), dtype=np.float32)
    
    # Generate X1 as variations of sin() with random multiplier, addition, and rotation
    for i in range(num_images):
        multiplier1 = np.random.uniform(0.8, 1.2)
        addition1 = np.random.uniform(-.1, 0.1)
        rotation1 = np.random.uniform(10, 80)  # Rotation angle in degrees
        
        for x in range(image_size):
            for y in range(image_size):
                X1[i, x, y] = multiplier1 * np.sin(rotation1 * np.pi / 180 + x * np.pi / 180) + addition1
    
    # Generate X2 as variations of cos() with random multiplier, addition, and rotation
    for i in range(num_images):
        multiplier2 = np.random.uniform(0.8, 1.2)
        addition2 = np.random.uniform(-.1, 0.1)
        rotation2 = np.random.uniform(10, 80)  # Rotation angle in degrees
        
        for x in range(image_size):
            for y in range(image_size):
                X2[i, x, y] = multiplier2 * np.cos(rotation2 * np.pi / 180 + y * np.pi / 180) + addition2
    
    # Calculate y as the element-wise product of X1 and X2
    y = X1 * X2
    
    # Convert data to PyTorch tensors
    X1 = torch.tensor(X1).unsqueeze(1)  # Add channel dimension
    X2 = torch.tensor(X2).unsqueeze(1)  # Add channel dimension
    y = torch.tensor(y).unsqueeze(1)    # Add channel dimension
    
    # Split into training and evaluation datasets
    X1_train = X1[:num_images_train]
    X2_train = X2[:num_images_train]
    y_train = y[:num_images_train]
    
    X1_eval = X1[num_images_train:]
    X2_eval = X2[num_images_train:]
    y_eval = y[num_images_train:]
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X1_train, X2_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    eval_dataset = TensorDataset(X1_eval, X2_eval, y_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, train_dataloader, eval_dataset, eval_dataloader
