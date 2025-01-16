# Load the best model
model.load_state_dict(torch.load("best_model.pth", weights_only=True))

# Evaluate on the test set
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Convert outputs to predicted labels using threshold (0.5 for each label)
        predicted = (torch.sigmoid(outputs) > 0.5).float()  # Convert outputs to probabilities and threshold
        
        # Calculate the number of correct predictions
        correct += (predicted == labels).sum().item()
        total += labels.numel()  # Total number of elements (all labels in the batch)

# Calculate average test loss and accuracy
test_loss /= len(test_dataloader)
accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
