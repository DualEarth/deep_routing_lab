# cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import image_size, hidden_size, kernel_size

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.hs = hidden_size
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(1, self.hs, kernel_size=self.kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.hs, self.hs, kernel_size=self.kernel_size)
        self.pool2 = nn.MaxPool2d(1, 1)
        self.fc1 = nn.Linear(self.hs * 2, self.hs)
        self.fc2 = nn.Linear(self.hs, self.image_size * self.image_size) 

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = self.pool2(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self.conv1(x2)
        x2 = F.relu(x2)
        x2 = self.pool1(x2)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)
        x2 = self.pool2(x2)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, self.image_size, self.image_size))  # Adjust according to your target shape
        return x
