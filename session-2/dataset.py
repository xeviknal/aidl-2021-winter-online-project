import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch


class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        self.transform = transform
        self.labels_csv = pd.read_csv(labels_path, sep=",")
        self.images_path = images_path

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.labels_csv.loc[idx, :]
        image_path = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, code - 1  # Try to run the whole model without this. It shouldn't change, right?
