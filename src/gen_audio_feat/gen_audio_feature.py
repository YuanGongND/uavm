# -*- coding: utf-8 -*-
# @Time    : 12/01/22 0:00 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : convnext.py

import argparse
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import models
import dataloader as dataloader
import torch
import numpy as np
from sklearn import metrics
from torch import nn

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def gen_save_feat(audio_model, val_loader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    with torch.no_grad():
        for i, (audio_input, labels, filename) in enumerate(val_loader):

            audio_input = audio_input.to(device)
            # compute output
            audio_output = audio_model.module.feature_extract(audio_input)
            predictions = audio_output.to('cpu').detach()
            #print(predictions.shape)

            for j in range(len(filename)):
                cur_filename = filename[j].split('/')[-1]
                cur_filename = save_path + '/' + cur_filename[:-5] + '.npy'
                np.save(cur_filename, predictions[j])
            print('processe {:d} of {:d}'.format(i*audio_output.shape[0], len(val_loader)*audio_output.shape[0]))

            exit()
    return 0

def gen_audio_feat(model, data, audio_conf, label_csv, num_class, batch_size=48, save_path='./'):
    print(model, data)

    # eval setting
    val_audio_conf = audio_conf
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data
    args.label_csv = label_csv
    args.exp_dir = './exp/dummy'
    args.loss_fn = torch.nn.BCELoss()
    val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

    audio_model = models.ConvNextOri(label_dim=num_class, model_id=2, pretrain=False, audioset_pretrain=False)
    sdA = torch.load(model, map_location=device)
    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sdA, strict=True)
    audio_model.eval()

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    gen_save_feat(audio_model, val_loader, save_path)

# for vggsound audio feature extraction
model = '/data/sls/scratch/yuangong/uavm/pretrained_models/feature_extractors/vgg_audio_feat.pth'
data = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/preprocess/train_vgg_convnext.json'
label_csv = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/preprocess/class_labels_indices_vgg.csv'
dataset = 'vggsound'
audio_conf = {'num_mel_bins': 128, 'target_length': 990, 'mean': -5.081, 'std': 4.4849, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset, 'teacher': True, 'noise': False, 'mode': 'evaluation'}
mAP = gen_audio_feat(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=309, batch_size=200, save_path='./test_vgg/')

# # for audioset audio feature extraction
# model = '/data/sls/scratch/yuangong/uavm/pretrained_models/feature_extractors/as_audio_feat.pth'
# data = '/data/sls/scratch/yuangong/avbyol/egs/audioset/preprocess/datafiles/balanced_train_data_type1_2_mean_dave_conv2_formal.json'
# label_csv = '/data/sls/scratch/yuangong/convast/egs/audioset/data/class_labels_indices.csv'
# dataset = 'audioset'
# audio_conf = {'num_mel_bins': 128, 'target_length': 1000, 'mean': -4.2677, 'std': 4.5690, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset, 'teacher': True, 'noise': False, 'mode': 'evaluation'}
# mAP = gen_audio_feat(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=527, batch_size=200, save_path='./test_as/')