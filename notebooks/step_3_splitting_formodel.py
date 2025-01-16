from sklearn.model_selection import train_test_split

# Split the balanced dataframe into training (80%) and remaining (20%)
train_df, remaining_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

# Split the remaining 20% into validation and test sets (50% of 20% = 10% for each)
val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)

# Create separate DataLoaders for training, validation, and test sets
train_dataset = RSNADataset(train_df, transform=transform)
val_dataset = RSNADataset(val_df, transform=transform)
test_dataset = RSNADataset(test_df, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Print the sizes of each set
print(f"Training set size: {len(train_dataloader.dataset)}")
print(f"Validation set size: {len(val_dataloader.dataset)}")
print(f"Test set size: {len(test_dataloader.dataset)}")
