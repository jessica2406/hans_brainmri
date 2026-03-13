import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BrainMRIDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the .npy file we created in Phase 1
        img = np.load(self.file_paths[idx])
        
        # Add a "channel" dimension (PyTorch expects [Channel, Height, Width])
        # Since our images are grayscale, Channel = 1
        img = torch.from_numpy(img).float().unsqueeze(0) 
        
        label = torch.tensor(self.labels[idx]).long()
        return img, label

def get_loaders(processed_dir, batch_size=16):
    categories = ['empyema', 'no_tumor', 'glioma', 'meningioma', 'pituitary']
    file_paths = []
    labels = []

    for idx, cat in enumerate(categories):
        cat_path = os.path.join(processed_dir, cat)
        for file in os.listdir(cat_path):
            file_paths.append(os.path.join(cat_path, file))
            labels.append(idx)

    # Split into Training (80%) and Validation (20%)
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = BrainMRIDataset(train_files, train_labels)
    val_ds = BrainMRIDataset(val_files, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, categories