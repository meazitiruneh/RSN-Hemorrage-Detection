from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize variables to store true labels and predictions
all_labels = []
all_predicted = []

# Evaluate on the test set
model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        
        # Apply sigmoid and threshold for multi-label outputs
        outputs = torch.sigmoid(outputs)
        predicted = (outputs > 0.5).float()  # Assuming threshold is 0.5

        # Store true labels and predictions (convert to numpy)
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_predicted = np.array(all_predicted)

# Calculate confusion matrix for each label
cm = confusion_matrix(all_labels.flatten(), all_predicted.flatten())

# Plot confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
