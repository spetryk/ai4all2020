import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import urllib

TRAIN_FRAC = 0.8


REAL_DOWNLOAD_URL = "https://ai4all-2020.s3.amazonaws.com/data/real_scans.pt"
FAKE_DOWNLOAD_URL = "https://ai4all-2020.s3.amazonaws.com/data/fake_scans.pt"

class MRIDataset(Dataset):
    def __init__(self, dataset_dir, transform=None,
                 train=True, download=True):

        self.transform = transform
        self.train = train

        real_scans_path = os.path.join(dataset_dir, "real_scans.pt")
        fake_scans_path = os.path.join(dataset_dir, "fake_scans.pt")
        if download:
            os.makedirs(dataset_dir, exist_ok=True)
            if not os.path.exists(real_scans_path):
                self.download(REAL_DOWNLOAD_URL, real_scans_path)
            if not os.path.exists(fake_scans_path):
                self.download(FAKE_DOWNLOAD_URL, fake_scans_path)

        real_scans = torch.load(real_scans_path) / 32768
        fake_scans = torch.load(fake_scans_path)
        self.n_real_scans = real_scans.size()[0]
        self.n_fake_scans = fake_scans.size()[0]

        self.scans = torch.cat([real_scans, fake_scans], dim=0)

        self.n_images = self.scans.size()[0]

        np.random.seed(42)
        indices = np.arange(self.n_images)
        np.random.shuffle(indices)

        split = int(TRAIN_FRAC * self.n_images)
        if self.train:
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]

    def __getitem__(self, idx):
        permuted_idx = self.indices[idx]
        scan = self.scans[permuted_idx].T.flip(0)[::2,::2]
        if self.transform is not None:
            scan = self.transform(scan)
        is_real = 1 - int(permuted_idx > self.n_real_scans)
        return scan, is_real

    def __len__(self):
        if self.train:
            return int(self.n_images * TRAIN_FRAC)
        else:
            return int(self.n_images * (1 - TRAIN_FRAC))

    def download(self, url, destination):
        urllib.request.urlretrieve(url, destination)
