import os
import pydicom
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class RSNADataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image path and labels
        img_path = self.dataframe.iloc[idx, 1]
        labels = self.dataframe.iloc[idx, 2:].values

        labels = labels.astype(np.float32)

        # Read DICOM image
        dicom_data = pydicom.dcmread(img_path)
        img = dicom_data.pixel_array

        # Convert to RGB if not already
        img = Image.fromarray(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply transformations (if any)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(labels, dtype=torch.float32)
