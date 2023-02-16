# -*- coding: utf-8 -*-
# @Time    : 5/21/22 2:08 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : visualize_rep.py

# for updated models. retrieval for a downsample of all dataset

import sys
import inspect
import argparse
import os
import torch
import numpy as np
from sklearn import metrics
from torch import nn
from numpy import dot
from numpy.linalg import norm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import dataloader_mm as dataloader
import models

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device is ' + str(device))

def get_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

# get mean
def get_sim_mat(a, b):
    B = a.shape[0]
    sim_mat = np.empty([B, B])
    for i in range(B):
        for j in range(B):
            sim_mat[i, j] = get_similarity(a[i, :], b[j, :])
    return sim_mat

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

def mae(y_true, predictions):
    return np.mean(np.abs(y_true - predictions))

def gen_att_map(audio_model, val_loader, exp_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    if os.path.exists('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap') == False:
        os.mkdir('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap')

    A_a_feat = [[],[],[],[],[],[]]
    A_v_feat = [[],[],[],[],[],[]]
    A_av_feat = [[], [], [], [], [], []]
    A_targets = []
    att_diff = []

    with torch.no_grad():
        for i, (audio_input, video_input, labels) in enumerate(val_loader):

            audio_input = audio_input.to(device)
            video_input = video_input.to(device)
            # compute output
            _, a_att_map, v_att_map, av_att_map = audio_model(audio_input, video_input)
            # all shared layer
            if v_att_map != None:
                for o_layer in range(len(a_att_map)):
                    A_a_feat[o_layer].append(a_att_map[o_layer])
                for o_layer in range(len(v_att_map)):
                    A_v_feat[o_layer].append(v_att_map[o_layer])
            # there should always be some shared layer
            for o_layer in range(len(av_att_map)):
                A_av_feat[o_layer].append(av_att_map[o_layer])

    if v_att_map != None:
        for o_layer in range(len(v_att_map)):
            A_a_feat[o_layer] = torch.cat(A_a_feat[o_layer])
            A_v_feat[o_layer] = torch.cat(A_v_feat[o_layer])
            A_a_feat[o_layer] = A_a_feat[o_layer].to('cpu').detach().numpy()
            A_v_feat[o_layer] = A_v_feat[o_layer].to('cpu').detach().numpy()
            print(A_a_feat[o_layer].shape, A_v_feat[o_layer].shape)
            print('a-v att difference {:f}'.format(mae(A_a_feat[o_layer], A_v_feat[o_layer])))
            np.save('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap/audio_attmap_{:d}.npy'.format(o_layer), A_a_feat[o_layer])
            np.save('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap/video_attmap_{:d}.npy'.format(o_layer), A_v_feat[o_layer])

    for o_layer in range(len(av_att_map)):
        A_av_feat[o_layer] = torch.cat(A_av_feat[o_layer])
        A_av_feat[o_layer] = A_av_feat[o_layer].to('cpu').detach().numpy()
        print(A_av_feat[o_layer].shape)
        np.save('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap/av_attmap_{:d}.npy'.format(o_layer), A_av_feat[o_layer])

    return 0

def get_eval(model, data, audio_conf, label_csv, num_class, model_type='cnn', batch_size=48, exp_name=None):
    print(model)
    print(data)

    # eval setting
    val_audio_conf = audio_conf
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data
    args.label_csv = label_csv
    args.exp_dir = './exp/dummy'
    args.loss_fn = torch.nn.BCELoss()

    val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

    if 'unified_trans' in model_type:
        depth_s = int(model_type.split('_')[-1])
        depth_i = int(model_type.split('_')[-2])
        s_embed_dim = int(model_type.split('_')[-3])
        embed_dim = int(model_type.split('_')[-4])
        print('independent depth is {:d} and shared depth is {:d}'.format(depth_i, depth_s))
        audio_model = models.FullAttTransVisual(embed_dim=embed_dim, num_heads=4, depth_i=depth_i, depth_s=depth_s, label_dim=num_class, a_seq_len=30, v_seq_len=30, a_input_dim=1024, v_input_dim=1024, s_embed_dim=s_embed_dim)
    sdA = torch.load(model, map_location=device)

    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sdA, strict=True)
    audio_model.eval()

    att_diff = gen_att_map(audio_model, val_loader, exp_name=exp_name)
    return att_diff

root_path = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'
exp_list = get_immediate_subdirectories(root_path)
exp_list.sort()
exp_list = ['testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1']
result = []
for exp_name in exp_list:
    try:
        exp_id = int(exp_name[6:8])
        exp_type = exp_name[4:6]
        if exp_id == 75 and exp_type == 'fm':
            print(exp_name)
            embed_dim=int(exp_name.split('-')[13][1:])
            sembed_dim = int(exp_name.split('-')[14][2:])
            d_i = int(exp_name.split('-')[16][2:])
            d_s = int(exp_name.split('-')[17][2:])
            print(embed_dim, sembed_dim, d_i, d_s)

            model = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'+exp_name+'/models/best_audio_model.pth'
            data = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/preprocess/datafiles/test_vgg_convnext_2.json'
            label_csv = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/preprocess/class_labels_indices_vgg.csv'
            dataset = 'vggsound'

            audio_conf = {'num_mel_bins': 128, 'target_length': 1000, 'freqm': 0, 'timem': 0, 'mixup': 0,
                          'dataset': dataset,
                          'mode': 'evaluation', 'mean': -4.2677, 'std': 4.5690, 'noise': False, 'student': False,
                          'feat_type': 'dummy', 'feat_norm': False,
                          'a_seq_len': 30, 'v_seq_len': 30, 'a_downsample': 1, 'v_downsample': 1, 'modality': 'av'}

            att_diff = get_eval(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=309,
                                        model_type='unified_trans_{:d}_{:d}_{:d}_{:d}'.format(embed_dim, sembed_dim, d_i, d_s), batch_size=400, exp_name=exp_name)

            result.append([exp_name] + att_diff)
            np.savetxt('./attdiff_summary_all_' + str(exp_id) + '_new.csv', result, delimiter=',', fmt='%s')

    except Exception as e:
        print(e)