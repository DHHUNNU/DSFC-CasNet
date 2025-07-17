# -----#
# author:HD
# year&month&day:2024:08:14
# -----#
###########度量#############
import torch
from sklearn import metrics


class PECN_metrics():
    def __init__(self):
        pass

    def multi_acc(self, pred, true):
        pred = pred.round()

        assert pred.shape == true.shape, f"Shape mismatch: pred.shape = {pred.shape}, true.shape = {true.shape}"

        correct = (pred == true).float()

        acc = correct.sum(dim=1) / correct.size(1)

        batch_acc = acc.mean()
        return batch_acc.item()

    def hamming_loss(self, pred, true):
        pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
        dpt = (pred - true).abs()
        return dpt.mean().item()


    def average_precision(self, pred, true):
        y_pred = pred.detach().numpy()
        y_true = true.numpy()
        ap = metrics.average_precision_score(y_true, y_pred)
        return ap
