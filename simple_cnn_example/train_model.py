# train_model.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn_model import SimpleCNN
from data_generation import generate_synthetic_data
from evaluate_model import evaluate_model
from config import batch_size

def train_model(model, dataset, batch_size=batch_size, num_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs1, inputs2, labels) in enumerate(dataloader):
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0