# -----#
# author:HD
# year&month&day:2025:01:05
# -----#
import torch
from torch.utils.data import Dataset
from scipy import io


class SCN_dataread(Dataset):

    def __init__(self, file_path, data_name, label_num):
        self.x, self.y = self.data_read(file_path, data_name, label_num)
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def data_read(self, file_path, data_name, label_num):
        data_path = file_path + data_name + '.mat'
        features_struct = io.loadmat(data_path)
        train_data = features_struct['data']

        label_start = train_data.shape[1] - label_num
        x = train_data[:, 0:label_start]
        y = train_data[:, label_start:2* label_start]

        return self.shape_transform(x, y)

    def shape_transform(self, x, y):

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.reshape((x.shape[0], 1, -1))
        return x, y


