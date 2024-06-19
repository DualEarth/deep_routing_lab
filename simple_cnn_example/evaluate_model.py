# evaluate_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (inputs1, inputs2, labels) in enumerate(dataloader):
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            if i == 0:
                plot_images_sample(inputs1[-1][-1].cpu().detach().numpy(), 
                                   inputs2[-1][-1].cpu().detach().numpy(), 
                                   outputs[-1][-1].cpu().detach().numpy(), 
                                   labels[-1][-1].cpu().detach().numpy())
    
    avg_loss = total_loss / len(dataloader)
    print(f'Average Evaluation Loss: {avg_loss:.4f}')

def plot_images_sample(input1, input2, output, label):
    fig, axes = plt.subplots(1, 4, figsize=(5, 3))
    images = [input1, input2, label, output]
    titles = ['Input 1', 'Input 2', 'Label', 'Output']
    
    for i in range(4):
        axes[i].imshow(images[i], cmap='gist_ncar')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('')  # Clear x-axis label
        axes[i].set_ylabel('')  # Clear y-axis label
        axes[i].tick_params(axis='both', which='both', length=0)  # Hide ticks
        
        # Add black outline for each subplot
        for spine in axes[i].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.show()