import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from src.model import SimpleCNN
from torch.utils.data import DataLoader
from src.dataset import RSNADataset
from torch import nn
import pandas as pd
from torchvision import transforms

# Load the best model
model = SimpleCNN().cuda()
model.load_state_dict(torch.load('best_model.pth'))

# Load the test dataset
test_df = pd.read_csv('../data/stage_2_test.csv')

# Apply necessary transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create test dataset and dataloader
test_dataset = RSNADataset(test_df, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate on the test set
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()  # Threshold for multi-label classification
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
