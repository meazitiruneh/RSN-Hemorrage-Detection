import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the model with 6 output classes (matching the trained model)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 500)
        self.fc2 = nn.Linear(500, 6)  # Change to 6 output classes
    
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Flatten the output for fully connected layer
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model with 6 output classes
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the saved model weights (which were trained for 6 subcategories)
model.load_state_dict(torch.load('final_model.pth', weights_only=True))
model.eval()

# Define the transform for image preprocessing (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Define the subcategories (for the 6 classes)
subcategories = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

# Definitions for each hemorrhage type
definitions = {
    'epidural': 'Epidural Hemorrhage: Bleeding between the skull and dura mater, often caused by trauma.',
    'intraparenchymal': 'Intraparenchymal Hemorrhage: Bleeding within brain tissue, often from high blood pressure or trauma.',
    'intraventricular': 'Intraventricular Hemorrhage: Bleeding in the brain\'s ventricles, often caused by trauma or premature birth.',
    'subarachnoid': 'Subarachnoid Hemorrhage: Bleeding in the space between the brain and the surrounding tissues, often due to aneurysm rupture.',
    'subdural': 'Subdural Hemorrhage: Bleeding between the dura mater and brain, often from head trauma.',
    'any': 'Any Hemorrhage: A general category for any type of brain hemorrhage.'
}

# Function to read and convert an image to a format usable by PIL
def load_image(image_path):
    file_extension = os.path.splitext(image_path)[-1].lower()
    
    if file_extension == '.dcm':  # If it's a DICOM file
        dicom_data = pydicom.dcmread(image_path)
        img_array = dicom_data.pixel_array  # Get pixel data from DICOM
        img = Image.fromarray(img_array).convert("RGB")  # Convert to RGB format for PIL
    else:  # If it's a regular image (e.g., jpg, png)
        img = Image.open(image_path).convert("RGB")
    
    return img

# Function to make a prediction
def predict_image(image_path):
    # Load and preprocess the image
    img = load_image(image_path)
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(img)
    
    # Get predicted probabilities (use sigmoid for multi-label classification)
    probabilities = torch.sigmoid(outputs)  # Sigmoid activation
    
    # Get the predicted classes (thresholded at 0.5)
    predicted_classes = (probabilities > 0.5).cpu().numpy().flatten()  # Threshold at 0.5

    # Map the predictions to the subcategories
    predictions = {subcategory: predicted_classes[i] for i, subcategory in enumerate(subcategories)}
    
    return predictions, probabilities, img

# Test with an example image (provide the image path here)
image_path = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/ID_000012eaf.dcm'  # Replace with your image path

# Make prediction
predictions, probabilities, img = predict_image(image_path)

# Print the results
print("Predictions (0 or 1 for each subcategory):")
for subcategory, prediction in predictions.items():
    print(f"{subcategory}: {prediction}")

# Optional: Print the probabilities for each subcategory
print("\nProbabilities for each subcategory:")
for subcategory, prob in zip(subcategories, probabilities.flatten()):
    print(f"{subcategory}: {prob:.4f}")

# Get the index of the highest probability, excluding 'any'
sorted_probabilities = probabilities.flatten().tolist()

# Find the maximum probability, excluding 'any'
max_prob_index = sorted_probabilities.index(max(sorted_probabilities))  # Find max without 'any'
if subcategories[max_prob_index] == 'any':
    # If 'any' has the highest probability, find the second highest
    sorted_probabilities[max_prob_index] = -1  # Set 'any' to a very low value
    max_prob_index = sorted_probabilities.index(max(sorted_probabilities))

# Get the subcategory and probability
max_subcategory = subcategories[max_prob_index]
max_probability = probabilities.flatten()[max_prob_index].item() * 100  # Convert to percentage

# Print conclusion for the maximum probability
print(f"\nConclusion: The model predicts '{max_subcategory}' with the highest probability of {max_probability:.2f}%.")
print(f"Scientific Description: {definitions[max_subcategory]}")

# Display the image (convert from Tensor to numpy array)
img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Remove batch dimension and convert to HxWxC format
img = (img * 0.5) + 0.5  # Denormalize

# Plot the image
plt.imshow(img)
plt.axis('on')  # Enable ticks
plt.title(f"Predicted: {max_subcategory} with probability {max_probability:.2f}%")
plt.show()
