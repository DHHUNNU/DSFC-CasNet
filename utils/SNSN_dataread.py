import torch
from torch.utils.data import Dataset
from scipy import io
import numpy as np


class SNSN_dataread(Dataset):
    def __init__(self, file_path, data_name, segment_type='fixed', segment_length=2048):

        self.segment_type = segment_type
        self.segment_length = segment_length
        self.x, self.y, self.z, self.x_factors = self.data_read(file_path, data_name)
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.z[item], self.x_factors[item]

    def normalize_signal(self, signal):

        p = 50
        x_normalized = []
        normalization_factors = []

        for i in range(signal.shape[0]):
            x_i = signal[i]

            # 对当前组数据进行归一化
            signal_min = np.min(x_i)
            signal_max = np.max(x_i)
            normalization_factor = (signal_max - signal_min) / 2

            normalized_signal = x_i  / normalization_factor * p

            x_normalized.append(normalized_signal)
            normalization_factors.append(normalization_factor)


        x_normalized = np.array(x_normalized)
        normalization_factors = np.array(normalization_factors).reshape(-1, 1)
        return x_normalized, normalization_factors

    def data_read(self, file_path, data_name):
        data_path = file_path + data_name + '.mat'

        features_struct = io.loadmat(data_path)

        train_data = features_struct['data']

        x = train_data[:, 0, :]
        y = train_data[:, 1, :]
        z = train_data[:, 2, :]
        p = 50
        x_normalized, x_factors = self.normalize_signal(x)
        z_normalized = z / x_factors * p

        x, y, z = self.segment_data(x_normalized, y, z_normalized)
        x, y, z = self.shape_transform(x, y, z)

        return x, y, z, x_factors


    def shape_transform(self, x, y, z):

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)


        x = x.reshape((x.shape[0], 1, -1))
        y = y.reshape((y.shape[0], -1))
        z = z.reshape((z.shape[0], -1))

        return x, y, z


    def segment_data(self, x, y, z):

        segments_x = []
        segments_y = []
        segments_z = []

        for i in range(x.shape[0]):
            x_i = x[i]
            y_i = y[i]
            z_i = z[i]

            if self.segment_type == 'fixed':
                x_seg, y_seg, z_seg, discarded_x, discarded_y, discarded_z = self.fixed_segmentation(x_i, y_i, z_i)
            elif self.segment_type == 'dynamic':
                x_seg, y_seg, z_seg = self.dynamic_segmentation(x_i, y_i, z_i)
            else:
                raise ValueError("Invalid segment_type. Choose either 'fixed' or 'dynamic'.")

            segments_x.append(x_seg)
            segments_y.append(y_seg)
            segments_z.append(z_seg)
        x_segments = torch.cat(segments_x, dim=0)
        y_segments = torch.cat(segments_y, dim=0)
        z_segments = torch.cat(segments_z, dim=0)

        return x_segments, y_segments, z_segments

    def fixed_segmentation(self, x, y, z):

        segments_x = []
        segments_y = []
        segments_z = []
        discarded_x = []
        discarded_y = []
        discarded_z = []

        for i in range(0, x.shape[0], self.segment_length):
            if i + self.segment_length <= x.shape[0]:
                segments_x.append(torch.from_numpy(x[i:i + self.segment_length]))
                segments_y.append(torch.from_numpy(y[i:i + self.segment_length]))
                segments_z.append(torch.from_numpy(z[i:i + self.segment_length]))
            else:

                discarded_x.append(torch.from_numpy(x[i:]))
                discarded_y.append(torch.from_numpy(y[i:]))
                discarded_z.append(torch.from_numpy(z[i:]))

        x_segments = torch.stack(segments_x)
        y_segments = torch.stack(segments_y)
        z_segments = torch.stack(segments_z)

        discarded_x = torch.cat(discarded_x) if discarded_x else None
        discarded_y = torch.cat(discarded_y) if discarded_y else None
        discarded_z = torch.cat(discarded_z) if discarded_z else None

        return x_segments, y_segments, z_segments, discarded_x, discarded_y, discarded_z

    def restore_fixed_segmented_signal(self, x_segments, y_segments, z_segments, discarded_x=None, discarded_y=None,
                                       discarded_z=None):

        restored_x = torch.cat(
            [x_segments.flatten(), discarded_x.flatten() if discarded_x is not None else torch.tensor([])])
        restored_y = torch.cat(
            [y_segments.flatten(), discarded_y.flatten() if discarded_y is not None else torch.tensor([])])
        restored_z = torch.cat(
            [z_segments.flatten(), discarded_z.flatten() if discarded_z is not None else torch.tensor([])])

        return restored_x, restored_y, restored_z

    def dynamic_segmentation(self, x, y, z):

        total_length = x.shape[0]
        padding_length = (self.segment_length - total_length % self.segment_length) % self.segment_length

        x_padded = np.pad(x, (0, padding_length), 'constant')
        y_padded = np.pad(y, (0, padding_length), 'constant')
        z_padded = np.pad(z, (0, padding_length), 'constant')

        segments_x = []
        segments_y = []
        segments_z = []

        for i in range(0, x_padded.shape[0], self.segment_length):
            segments_x.append(torch.from_numpy(x_padded[i:i + self.segment_length]))  # 转为PyTorch tensor
            segments_y.append(torch.from_numpy(y_padded[i:i + self.segment_length]))
            segments_z.append(torch.from_numpy(z_padded[i:i + self.segment_length]))

        x_segments = torch.stack(segments_x)
        y_segments = torch.stack(segments_y)
        z_segments = torch.stack(segments_z)

        return x_segments, y_segments, z_segments







