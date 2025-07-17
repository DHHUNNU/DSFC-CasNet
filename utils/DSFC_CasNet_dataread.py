# -----#
# author:HD
# year&month&day:2024:08:14
# -----#
import torch
from torch.utils.data import Dataset
import scipy.io as io

class NLNet_dataread(Dataset):

    def __init__(self, file_path, data_name):
        self.x, self.padding_len, self.norm_factors = self.data_read(file_path, data_name)
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item]

    def data_read(self, file_path, data_name):
        data_path = file_path + data_name + '.mat'
        features_struct = io.loadmat(data_path)
        train_data = features_struct['data']
        x = torch.from_numpy(train_data)
        x_flat = x.squeeze(0)

        block_size = 2048
        total_len = x_flat.shape[0]
        n_blocks = total_len // block_size
        res_len = total_len % block_size

        x_main = x_flat[:n_blocks * block_size].reshape(n_blocks, block_size)
        x_res = x_flat[n_blocks * block_size:]
        padding_len = block_size - res_len
        x_res_padded = torch.nn.functional.pad(x_res, (0, padding_len))

        x_all = torch.cat([x_main, x_res_padded.unsqueeze(0)], dim=0)
        x, norm_factors = self.max_abs_normalize(x_all)
        return self.shape_transform(x), padding_len, norm_factors

    def max_abs_normalize(self, x, eps=1e-8):
        norm_factors = x.abs().max(dim=1, keepdim=True).values
        norm_factors[norm_factors == 0] = eps
        x = x / norm_factors
        return x, norm_factors

    def shape_transform(self, x):
        return x.reshape((x.shape[0], 1, -1))

    @staticmethod
    def restore_sequence_static(output_segments, padding_len):
        if output_segments.dim() == 3:
            output_segments = output_segments.squeeze(1)
        flat = output_segments.reshape(-1)
        if padding_len > 0:
            recovered = flat[:-padding_len]
        else:
            recovered = flat
        return recovered.unsqueeze(0)

class SegmentationModeSelector:
    def __init__(self, segment_type='fixed', segment_length=2048):
        self.segment_type = segment_type
        self.segment_length = segment_length

    def segment(self, x, y):
        if self.segment_type == 'fixed':
            return self.fixed_segmentation(x, y)
        elif self.segment_type == 'dynamic':
            return self.dynamic_segmentation(x, y)
        else:
            raise ValueError("segment_type must be 'fixed' or 'dynamic'")

    def fixed_segmentation(self, x, y):
        total_len = x.shape[0]
        n_blocks = total_len // self.segment_length

        x_segs = x[:n_blocks * self.segment_length].reshape(n_blocks, self.segment_length)
        y_segs = y[:n_blocks * self.segment_length].reshape(n_blocks, self.segment_length)
        pad_lengths = torch.zeros((n_blocks, 1), dtype=torch.long)

        return x_segs, y_segs, pad_lengths

    def dynamic_segmentation(self, x, y):
        seg_x, seg_y, pad_lengths = [], [], []
        i = 0
        while i < len(x):
            window_start = i
            window_end = i + 1500
            if window_end >= len(x):
                window_end = len(x)

            while window_end < len(y) and y[window_end - 1] == 1:
                window_end += 1
                if window_end - window_start >= self.segment_length:
                    break

            seg_len = window_end - window_start
            x_seg = x[window_start:window_end]
            y_seg = y[window_start:window_end]

            if seg_len < self.segment_length:
                pad_len = self.segment_length - seg_len
                x_seg = torch.nn.functional.pad(x_seg, (0, pad_len))
                y_seg = torch.nn.functional.pad(y_seg, (0, pad_len))
            else:
                x_seg = x_seg[:self.segment_length]
                y_seg = y_seg[:self.segment_length]
                pad_len = 0

            seg_x.append(x_seg)
            seg_y.append(y_seg)
            pad_lengths.append(pad_len)
            i = window_end

        x_segments = torch.stack(seg_x)
        y_segments = torch.stack(seg_y)
        pad_lengths = torch.tensor(pad_lengths).view(-1, 1)

        return x_segments, y_segments, pad_lengths

    @staticmethod
    def restore_sequence(output_segments, pad_lengths, norm_factors):

        output_segments = output_segments.squeeze(1)
        output_denorm = output_segments * norm_factors

        restored = []
        for i, seg in enumerate(output_denorm):
            if pad_lengths[i] > 0:
                restored.append(seg[:-pad_lengths[i]])
            else:
                restored.append(seg)

        full = torch.cat(restored).unsqueeze(0)
        return full

class SNSNet_dataread(Dataset):
    def __init__(self, file_path, data_name):
        self.x = self.data_read(file_path, data_name)
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx]

    def data_read(self, file_path, data_name):
        data_path = file_path + data_name + '.mat'
        mat_struct = io.loadmat(data_path)
        raw_data = mat_struct['data']
        x = torch.from_numpy(raw_data)
        return x

    @staticmethod
    def max_abs_normal(x, eps=1e-8):
        norm_factors = x.abs().max(dim=1, keepdim=True).values
        norm_factors[norm_factors == 0] = eps
        x_normalized = x / norm_factors
        return x_normalized, norm_factors
