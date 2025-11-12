import os
import torch
from torch.utils.data import Dataset
import scipy.io


class CustomDataset(Dataset):
    def __init__(self, pa_data_dir, ground_truth_dir, transform=None):
        self.pa_data_dir = pa_data_dir
        self.ground_truth_dir = ground_truth_dir
        self.mat_files = sorted(os.listdir(pa_data_dir))
        self.img_files = sorted(os.listdir(ground_truth_dir))
        self.transform = transform

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_data = scipy.io.loadmat(os.path.join(self.pa_data_dir, self.mat_files[idx]))["sensor_data_2D"]
        mat_data = torch.tensor(mat_data, dtype=torch.float32).unsqueeze(0)
        img_path = os.path.join(self.ground_truth_dir, self.img_files[idx])
        img = scipy.io.loadmat(img_path)["p0"]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return mat_data, img
