import pandas as pd
import os
import pydicom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torch import nn
import torch.optim as optim
from PIL import Image
import numpy as np

# Load the dataset
df = pd.read_csv('/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv')

# Define the train directory path
train_path = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'

# List all DICOM file names from the directory
CT_image_file_names = os.listdir(train_path)

# Filter the dataset for hemorrhage cases (Label == 1)
df = df[df['Label'] == 1]

# Function to split the ID column manually
def split_id(id_value):
    parts = id_value.split('_', 2)
    return parts[1], parts[2] if len(parts) > 2 else 'N/A'

# Apply the function to split the ID and create new columns
df[['File', 'Subcategory']] = df['ID'].apply(split_id).apply(pd.Series)

# Drop the 'ID' column
df = df.drop(columns=['ID'])

# Rearrange the columns and rename them for clarity
df = df[['File', 'Subcategory', 'Label']]
df.columns = ['File Name', 'Subcategory', 'Label']

# Add 'ID_' prefix to the 'File Name' column
df['File Name'] = 'ID_' + df['File Name']

# Define subcategories
subcategories = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

# Create binary columns for each subcategory
for subcategory in subcategories:
    df[subcategory] = df.apply(lambda row: 1 if row['Subcategory'] == subcategory and row['Label'] == 1 else 0, axis=1)

# Drop 'Subcategory' and 'Label' columns
df = df.drop(columns=['Subcategory', 'Label'])

# Aggregate by 'File Name', taking the maximum value for each subcategory column
df_aggregated = df.groupby('File Name').max().reset_index()

# Save the 'File Name' values into a list
file_names_list = df_aggregated['File Name'].tolist()

# Create a DataFrame for DICOM image paths
ICH_dicom_image_paths = pd.DataFrame({
    'File Name': [n for n in file_names_list],
    'image_path': [train_path + n + '.dcm' for n in file_names_list]
})

# Merge the image paths with the aggregated data
merged_df = pd.merge(left=ICH_dicom_image_paths, right=df_aggregated, on='File Name', how='inner')

# --- Equal Sampling from Each Subcategory ---

# Ensure there are 100 images from each subcategory
balanced_df = pd.DataFrame()

for subcategory in subcategories:
    # Filter the dataset by subcategory
    subcategory_df = merged_df[merged_df[subcategory] == 1]
    
    # Sample 100 images for this subcategory
    sampled_df = subcategory_df.sample(n=100, random_state=42)  # Adjust random_state for reproducibility
    
    # Append to the balanced dataframe
    balanced_df = pd.concat([balanced_df, sampled_df])

# Shuffle the final balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the shape of the balanced dataset (should be 600 images)
print(f"Balanced dataset shape: {balanced_df.shape}")

# Save the balanced DataFrame
balanced_df.to_csv('/kaggle/working/balanced_out.csv', index=False)
