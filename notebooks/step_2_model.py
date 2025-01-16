import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
from PIL import Image
import numpy as np
from torchvision import transforms

# --- Custom Dataset Class ---
class RSNADataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 1]
        labels = self.dataframe.iloc[idx, 2:].values  # Get binary labels for all subcategories
        
        # Convert labels to float32
        labels = labels.astype(np.float32)

        # Load the DICOM image using pydicom
        dicom_data = pydicom.dcmread(img_path)
        img = dicom_data.pixel_array
        
        # Convert image to PIL Image
        img = Image.fromarray(img)
        
        # Convert single-channel image to 3-channel image (if needed)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)
        
        # Return image and labels
        return img, torch.tensor(labels, dtype=torch.float32)

# --- Define Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# --- Create Dataset and DataLoader ---
dataset = RSNADataset(balanced_df, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --- Define the CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define the convolutional layers and fully connected layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Input: 3 channels, Output: 16 channels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Input: 16 channels, Output: 32 channels
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer with 2x2 kernel
        
        # Calculate the output size after convolutions and pooling
        # 224x224 -> 112x112 -> 56x56 after pooling
        self.fc1 = nn.Linear(32 * 56 * 56, 500)  # Flattened image to 500 nodes
        self.fc2 = nn.Linear(500, 6)  # Output layer for 6 subcategories (multi-label classification)

    def forward(self, x):
        # Convolutional and pooling layers with ReLU activation
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 56 * 56)  # Adjust this if the input image size changes
        
        # Fully connected layers with ReLU activation
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)  # Output layer (raw logits)
        
        return x

# --- Initialize Model, Optimizer, and Loss Function ---
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Use BCEWithLogitsLoss for multi-label classification
criterion = nn.BCEWithLogitsLoss()

# --- Training Loop ---
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss for multi-label classification
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

print("Training complete!")
