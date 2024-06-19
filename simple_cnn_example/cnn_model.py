# cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import image_size, hidden_size, kernel_size, verbose_and_break

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.hs = hidden_size
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(1, self.hs, kernel_size=self.kernel_size)  # Input: (1, 8, 8), Output: (self.hs, 6, 6)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (self.hs, self.kernel_size, self.kernel_size)
        self.conv2 = nn.Conv2d(self.hs, self.hs, kernel_size=self.kernel_size)  # Output: (self.hs, 1, 1)
        self.pool2 = nn.MaxPool2d(1, 1)  # Adjusted pool2 to avoid size reduction issue
        self.fc1 = nn.Linear(self.hs * 2, self.hs)  # Adjusted for concatenated tensor size
        self.fc2 = nn.Linear(self.hs, self.image_size * self.image_size)  # Adjusted for target size

    def forward(self, x1, x2):
        # Process x1 through conv layers
        x1 = self.conv1(x1)
        if verbose_and_break: 
            print(f'After conv1 for x1: {x1.size()}')

        x1 = F.relu(x1)
        if verbose_and_break: 
            print(f'After ReLU for x1: {x1.size()}')
            
        x1 = self.pool1(x1)
        if verbose_and_break: 
            print(f'After pool1 for x1: {x1.size()}')

        x1 = self.conv2(x1)
        if verbose_and_break: 
            print(f'After conv2 for x1: {x1.size()}')
            
        x1 = F.relu(x1)
        if verbose_and_break: 
            print(f'After ReLU for x1: {x1.size()}')
            
        x1 = self.pool2(x1)
        if verbose_and_break: 
            print(f'After pool2 for x1: {x1.size()}')

        x1 = torch.flatten(x1, 1)
        if verbose_and_break: 
            print(f'Flattened x1: {x1.size()}')

        # Process x2 through conv layers (same layers as x1)
        x2 = self.conv1(x2)
        if verbose_and_break: 
            print(f'After conv1 for x2: {x2.size()}')
            
        x2 = F.relu(x2)
        if verbose_and_break: 
            print(f'After ReLU for x2: {x2.size()}')
            
        x2 = self.pool1(x2)
        if verbose_and_break: 
            print(f'After pool1 for x2: {x2.size()}')

        x2 = self.conv2(x2)
        if verbose_and_break: 
            print(f'After conv2 for x2: {x2.size()}')
            
        x2 = F.relu(x2)
        if verbose_and_break: 
            print(f'After ReLU for x2: {x2.size()}')
            
        x2 = self.pool2(x2)
        if verbose_and_break: 
            print(f'After pool2 for x2: {x2.size()}')

        x2 = torch.flatten(x2, 1)
        if verbose_and_break: 
            print(f'Flattened x2: {x2.size()}')

        # Concatenate x1 and x2
        x = torch.cat((x1, x2), dim=1)
        if verbose_and_break: 
            print(f'Concatenated tensor: {x.size()}')

        # Fully connected layers
        x = self.fc1(x)
        if verbose_and_break: 
            print(f'After fc1: {x.size()}')
            
        x = F.relu(x)
        x = self.fc2(x)
        if verbose_and_break: 
            print(f'After fc2: {x.size()}')

        # Reshape x to match the shape of labels (if necessary)
        x = torch.reshape(x, (-1, 1, self.image_size, self.image_size))  # Adjust according to your target shape
        if verbose_and_break: 
            print(f'Reshaped x: {x.size()}')
            print(thiswillstopthenotebookfromrunning)

        return x
