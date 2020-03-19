from torch.utils.data import Dataset
import numpy as np


class AdvImgDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform

        self.data = np.load(path + "/advimg_images.npy")
        self.labels = np.load(path + "/advimg_labels.npy")

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        image = self.data[idx, :, :, :]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
