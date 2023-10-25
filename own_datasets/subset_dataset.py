import random

import torch
import numpy as np


class SingleClassSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, target_class, max_num_samples=None):
        self.dataset = dataset
        self.indices = np.where(np.array(dataset.targets) == target_class)[0]
        self.targets = np.array(dataset.targets)[self.indices]
        self.target_class = target_class

        if max_num_samples is not None and len(self.indices) >= max_num_samples:
            self.indices = np.array(random.sample(self.indices.tolist(), k=max_num_samples))
            self.targets = np.array(dataset.targets)[self.indices]

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)