# Training loop with validation
num_epochs = 5
best_val_loss = float('inf')  # Initialize the best validation loss

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training phase
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()  # Switch model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No gradients needed for validation
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Calculate average loss for the epoch
    train_loss = running_loss / len(train_dataloader)
    val_loss = val_loss / len(val_dataloader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save model checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model checkpoint saved!")
