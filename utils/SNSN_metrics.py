import torch
import torch.nn.functional as F

class SNSN_metrics():
    def __init__(self):
        pass

    def MSE(self, pred, true):

        mse = F.mse_loss(pred, true)
        return mse.item()

    def PSNR(self, pred, true):

        mse = self.MSE(pred, true)
        if mse == 0:
            return float('inf')

        max_pixel = max(pred.max().item(), true.max().item())


        psnr = 20 * torch.log10(max_pixel / torch.sqrt(torch.tensor(mse)))
        return psnr.item()

    def SNR(self, pred, true):

        signal_power = torch.mean(true ** 2)


        noise_power = torch.mean((pred - true) ** 2)

        if noise_power == 0:
            return float('inf')


        snr = 10 * torch.log10(signal_power / noise_power)

        return snr.item()

    def NCC(self, pred, true):

        pred_mean = pred.mean()
        true_mean = true.mean()

        ncc_num = ((pred - pred_mean) * (true - true_mean)).sum()

        ncc_denom = torch.sqrt(((pred - pred_mean) ** 2).sum() * ((true - true_mean) ** 2).sum())

        ncc = ncc_num / ncc_denom
        return ncc.item()

    def dynamic_segmentation(self, amt_signal, amt_encoding):

        window_size = 1500
        step_size = 1
        start = 0
        segmented_signals = []
        segmented_labels = []
        padding_values = []

        while start < len(amt_signal):
            end = start + window_size

            while end < len(amt_encoding) and amt_encoding[end - 1] == 1:
                end += step_size

            segment = amt_signal[start:end]
            label_segment = amt_encoding[start:end]

            if len(segment) > 2048:
                segment = segment[:2048]
                label_segment = label_segment[:2048]

            padding = 2048 - len(segment)
            segment = torch.cat((segment, torch.zeros(padding)))
            label_segment = torch.cat((label_segment, torch.zeros(padding)))

            segmented_signals.append(segment)
            segmented_labels.append(label_segment)
            padding_values.append(padding)

            start = end

        segmented_signals = torch.stack(segmented_signals)
        segmented_labels = torch.stack(segmented_labels)
        padding_values = torch.tensor(padding_values)

        return segmented_signals, segmented_labels, padding_values

    def fixed_segmentation(self, x):

        self.segment_length = 2048
        segments_x = []
        discarded_x = None

        if len(x.shape) == 3:
            x = x.squeeze(1)

        total_length = x.shape[0] * self.segment_length
        num_segments = x.shape[0]

        remainder = total_length % self.segment_length

        for i in range(num_segments):
            signal = x[i]
            segments_x.append(signal)

        if remainder != 0:
            discarded_x = x[-1, remainder:]

        x_segments = torch.stack(segments_x) if segments_x else None

        return x_segments, discarded_x

    def restore_and_denormalize_signal(self, x_segments, discarded_x=None, normalization_factors=None):
        if x_segments is None or x_segments.numel() == 0:
            return torch.tensor([])

        restored_x = x_segments.flatten()

        if discarded_x is not None:
            restored_x = torch.cat([restored_x, discarded_x.flatten()])

        if normalization_factors is not None:
            normalization_factor = normalization_factors[0]
            restored_x = restored_x * normalization_factor / 50

        return restored_x