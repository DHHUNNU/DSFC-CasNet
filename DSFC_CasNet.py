# -----#
# author:HD
# year&month&day:2025:02:14
# -----#
import torch
import numpy as np
from utils.DSFC_CasNet_dataread import NLNet_dataread
from utils.DSFC_CasNet_dataread import SegmentationModeSelector
from utils.DSFC_CasNet_dataread import SNSNet_dataread
from network.SCN import SCN
from network.SNSN import SNSN

def NLNet_test(model_name, data_name):
    model_path = './model/' + model_name
    data_path = './datasets/test_datasets/'
    net = SCN(9, 256)
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    net.eval()
    dataset = NLNet_dataread(data_path, data_name)
    input_ = dataset.x
    padd_len = dataset.padding_len
    norm_factors = dataset.norm_factors
    with torch.no_grad():
        output_ = net(input_)
    output_ = torch.where(output_ > 0.5, torch.ones_like(output_), torch.zeros_like(output_))
    out = output_
    return out, padd_len, norm_factors, input_

model_name = 'NLNet.pth'
data_name = 'EL22208ATS4EX'
output, padding_len, norm_fac, input_ = NLNet_test(model_name, data_name)
pre_label = NLNet_dataread.restore_sequence_static(output, padding_len)
input = NLNet_dataread.restore_sequence_static(input_, padding_len)

def SNSN_test(model_name, data_name, label):
    model_path = './model/' + model_name
    data_path = './datasets/test_datasets/'
    net = SNSN(9, 128)
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    net.eval()
    dataset = SNSNet_dataread(data_path, data_name)
    input = dataset.x
    input = input.squeeze()
    label = label.squeeze()
    segmenter = SegmentationModeSelector(segment_type='dynamic', segment_length=2048)
    seg_input, seg_label, pad_len = segmenter.segment(input, label)
    seg_normal_input, norm_factors = SNSNet_dataread.max_abs_normal(seg_input)
    seg_normal_input = seg_normal_input.unsqueeze(1)
    seg_label = seg_label.squeeze(1)
    p = 50
    with torch.no_grad():
        output_ = net(seg_normal_input, seg_label)/p

    out = SegmentationModeSelector.restore_sequence(output_, pad_len, norm_factors)
    return out, input.view(1,-1)

model_name2 = 'SNSNet.pth'
final_out, input = SNSN_test(model_name2, data_name, pre_label)
torch.set_printoptions(profile="full")

np.savetxt("output.csv", final_out, delimiter=",")
np.savetxt("input.csv", input, delimiter=",")

