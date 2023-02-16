# -*- coding: utf-8 -*-
# @Time    : 5/21/22 2:08 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : visualize_rep.py

# for updated models. retrieval for a downsample of all dataset, return R@1

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

def eval(audio_model, val_loader, exp_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    if os.path.exists('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat') == False:
        os.mkdir('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat')
    if os.path.exists('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/simmat') == False:
        os.mkdir('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/simmat')

    retrieval_recall = []
    for out_layer in range(0, 8):

        if os.path.exists('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(out_layer)) == False:
            os.mkdir('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(out_layer))

        A_a_feat = []
        A_v_feat = []
        A_targets = []

        with torch.no_grad():
            for i, (audio_input, video_input, labels) in enumerate(val_loader):

                audio_input = audio_input.to(device)
                video_input = video_input.to(device)
                # compute output
                audio_output = audio_model.module.extract_feat(audio_input, 'a', out_layer)
                video_output = audio_model.module.extract_feat(video_input, 'v', out_layer)

                # pool the embedding through the time if intermediate rep
                if audio_output.dim() == 3:
                    audio_output = torch.mean(audio_output, dim=1)
                if video_output.dim() == 3:
                    video_output = torch.mean(video_output, dim=1)

                A_a_feat.append(audio_output)
                A_v_feat.append(video_output)
                A_targets.append(labels)

        A_a_feat = torch.cat(A_a_feat)
        A_v_feat = torch.cat(A_v_feat)
        A_targets = torch.cat(A_targets)

        A_a_feat = A_a_feat.to('cpu').detach().numpy()
        A_v_feat = A_v_feat.to('cpu').detach().numpy()

        print(A_a_feat.shape, A_v_feat.shape)

        downsample = 1
        sim_mat = get_sim_mat(A_a_feat[::downsample, :], A_v_feat[::downsample, :])
        result = compute_metrics(sim_mat)
        print_computed_metrics(result)
        retrieval_recall.append(result['R10'])

        if A_a_feat.shape[1] == 309:
            acc_a = metrics.accuracy_score(np.argmax(A_targets, 1), np.argmax(A_a_feat, 1))
            acc_v = metrics.accuracy_score(np.argmax(A_targets, 1), np.argmax(A_v_feat, 1))
            acc_av = metrics.accuracy_score(np.argmax(A_targets, 1), np.argmax(A_a_feat + A_v_feat, 1))
            print(acc_a, acc_v, acc_av)

        # if need to save the feature
        #np.save('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(out_layer) + '/audio_rep_{:s}.npy'.format(str(out_layer)), A_a_feat)
        #np.save('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(out_layer) + '/video_rep_{:s}.npy'.format(str(out_layer)), A_v_feat)

    return retrieval_recall

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
        audio_model = models.UnifiedTrans(embed_dim=embed_dim, num_heads=4, depth_i=depth_i, depth_s=depth_s, label_dim=num_class, a_seq_len=30, v_seq_len=30, a_input_dim=1024, v_input_dim=1024, s_embed_dim=s_embed_dim)
    sdA = torch.load(model, map_location=device)

    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sdA, strict=True)
    audio_model.eval()

    acc = eval(audio_model, val_loader, exp_name=exp_name)
    print(acc)
    return acc

root_path = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'
exp_name_list = [
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se32-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se512-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di0-ds6-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
             'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di6-ds0-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

exp_list = []
for exp_name in exp_name_list:
    for r in ['-r1', '-r2', '-r3']:
        exp_list.append(exp_name + r)
print(exp_list)
result = []
for exp_name in exp_list:
    try:
        exp_id = int(exp_name[6:8])
        exp_type = exp_name[4:6]

        print(exp_name)
        embed_dim=int(exp_name.split('-')[13][1:])
        sembed_dim = int(exp_name.split('-')[14][2:])
        d_i = int(exp_name.split('-')[16][2:])
        d_s = int(exp_name.split('-')[17][2:])
        print(embed_dim, sembed_dim, d_i, d_s)

        model = '/data/sls/scratch/yuangong/uavm/pretrained_models/retrieval/'+exp_name+'.pth'
        data = './test_vgg_convnext_2_5_per_class.json'
        label_csv = './class_labels_indices_vgg.csv'
        dataset = 'vggsound'

        audio_conf = {'num_mel_bins': 128, 'target_length': 1000, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                      'mode':'evaluation', 'mean': -4.2677, 'std': 4.5690, 'noise': False, 'feat_norm': False,
                      'a_seq_len': 30, 'v_seq_len': 30, 'modality': 'av'}

        retrieval_recall = get_eval(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=309, model_type='unified_trans_{:d}_{:d}_{:d}_{:d}'.format(embed_dim, sembed_dim, d_i, d_s), batch_size=400, exp_name=exp_name)
        result.append([exp_name] + retrieval_recall)
        np.savetxt('./retrieval_summary_final_r10.csv', result, delimiter=',', fmt='%s')

    except Exception as e:
        print(e)