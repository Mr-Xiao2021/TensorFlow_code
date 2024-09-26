import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MySegmentationDataset(Dataset):
    def __init__(self, images_path: list, labels_path: list, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx]).convert("RGB")
        label = Image.open(self.labels_path[idx]).convert("L")

        if self.transform is not None:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label)).long()

        return image, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        # Stack the images and labels to form a batch
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels