# -*- coding: utf-8 -*-
# @Time    : 12/25/21 3:45 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : convnext.py

# pure convnet (convnext model), used for audio branch feature training

import torch.nn as nn
import torch
import torchvision
from torch.cuda.amp import autocast
from torchvision.models.feature_extraction import create_feature_extractor

# convnext as audio feature extractor
class ConvNextOri(nn.Module):
    def __init__(self, label_dim=309, pretrain=True, model_id=0, audioset_pretrain=False):
        super().__init__()
        print('now train a convnext model ' + str(model_id))
        model_id = int(model_id)
        if model_id == 0:
            self.model = torchvision.models.convnext_tiny(pretrained=pretrain)
        elif model_id == 1:
            self.model = torchvision.models.convnext_small(pretrained=pretrain)
        elif model_id == 2:
            self.model = torchvision.models.convnext_base(pretrained=pretrain)
        elif model_id == 3:
            self.model = torchvision.models.convnext_large(pretrained=pretrain)
        hid_dim = [768, 768, 1024, 1536]
        self.model = torch.nn.Sequential(*list(self.model.children()))
        self.model[-1][-1] = torch.nn.Linear(hid_dim[model_id], label_dim)

        new_proj = torch.nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4), bias=True)
        print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model[0][0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model[0][0][0].bias
        self.model[0][0][0] = new_proj

    @autocast()
    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        out = self.model(x)
        return out

    def feature_extract(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.model[:-2](x)
        x = torch.mean(x, dim=2)
        x = x.transpose(1, 2)
        return x