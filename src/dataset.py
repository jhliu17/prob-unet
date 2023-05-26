import torch
import numpy as np

from torch.utils.data import Dataset


class BoeChiuFluidSegDataset(Dataset):
    def __init__(self, npz_path: str):
        super().__init__()
        self.npz = np.load(npz_path)
        self.x = self.npz['x']
        self.y = self.npz['y']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        tensor_x = torch.from_numpy(self.x[index]).float()
        tensor_y = torch.from_numpy(self.y[index]).long()
        return tensor_x, tensor_y
