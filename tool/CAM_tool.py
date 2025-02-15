import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

def make_F2_feature(outputs):
    # multi-scale fusion
    F2feature_list = [output[1].cpu() for output in outputs]
    F2_size = F2feature_list[0].size()[2:]
    F2feature_list = [F.interpolate(o, F2_size, mode='bilinear', align_corners=False) for o in F2feature_list]
    F2Feature = torch.sum(torch.stack(F2feature_list, 0), 0) / 4
    return F2Feature
