import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import RSNADataset
from src.model import SimpleCNN
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Load and preprocess the dataset
df = pd.read_csv('../data/stage_2_train.csv')

# Apply any necessary data transformations (e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Create dataset and dataloaders
train_dataset = RSNADataset(train_df, transform=transform)
val_dataset = RSNADataset(val_df, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, optimizer, and loss function
model = SimpleCNN().cuda()  # Assuming you have a CUDA-compatible GPU
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader):.4f}")

    # Validate after each epoch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_dataloader):.4f}")

    # Optionally, save the model with the best validation performance
    torch.save(model.state_dict(), 'best_model.pth')
