from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize lists to store predicted and true labels
predicted_labels = []
true_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        # Get the model's outputs
        outputs = model(images)
        
        # Apply sigmoid and threshold (for multi-label classification)
        outputs = torch.sigmoid(outputs)
        predicted = (outputs > 0.5).float()  # Threshold at 0.5
        
        # Collect the predicted and true labels (as lists)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

# Calculate Precision, Recall, F1 Score for each label (macro-average)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
