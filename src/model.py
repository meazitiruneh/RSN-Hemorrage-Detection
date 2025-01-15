import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Define CNN layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 500)
        self.fc2 = nn.Linear(500, 6)  # 6 outputs for multi-label classification

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 56 * 56)

        # Fully connected layers with ReLU activation
        x = nn.ReLU()(self.fc1(x))

        # Output layer (multi-label classification)
        x = self.fc2(x)
        return x
