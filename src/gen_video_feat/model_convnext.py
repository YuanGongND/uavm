# code modified from https://github.com/antoine77340/video_feature_extractors

import sys
import torch as th
import torchvision.models as models
from videocnn.models import resnext
from torch import nn


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model(args):
    assert args.type in ['2d', '3d']
    if args.type == '2d':
        if args.model_size == 0:
            print('Loading ConvNext Tiny')
            model = models.convnext_tiny(pretrained=True)
        elif args.model_size == 1:
            print('Loading ConvNext Small')
            model = models.convnext_small(pretrained=True)
        elif args.model_size == 2:
            print('Loading ConvNext Base')
            model = models.convnext_base(pretrained=True)
        elif args.model_size == 3:
            print('Loading ConvNext Large')
            model = models.convnext_large(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        model = model.cuda()
    else:
        print('Loading 3D-ResneXt-101 ...')
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False)
        model = model.cuda()
        model_data = th.load(args.resnext101_model_path)
        model.load_state_dict(model_data)

    model.eval()
    print('loaded')
    return model
