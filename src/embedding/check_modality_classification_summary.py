# -*- coding: utf-8 -*-
# @Time    : 5/21/22 2:08 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : check_modality_classification2.py

# for updated models.

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
#from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def cal_modality_cla(a_feat, v_feat):
    sample_num = int(a_feat.shape[0] / 2)

    a_label = np.array([0] * a_feat.shape[0]).reshape(a_feat.shape[0], 1)
    v_label = np.array([1] * v_feat.shape[0]).reshape(v_feat.shape[0], 1)

    a_feat = np.concatenate((a_feat, a_label), axis=1)
    v_feat = np.concatenate((v_feat, v_label), axis=1)

    tr_a_samples = a_feat[:sample_num, :]
    tr_v_samples = v_feat[:sample_num, :]
    te_a_samples = a_feat[sample_num:, :]
    te_v_samples = v_feat[sample_num:, :]

    tr_samples = np.concatenate((tr_a_samples, tr_v_samples), axis=0)
    te_samples = np.concatenate((te_a_samples, te_v_samples), axis=0)

    np.random.shuffle(tr_samples)

    reg = LogisticRegression(penalty='l1', solver='liblinear', random_state=0).fit(tr_samples[:, :-1], tr_samples[:, -1])
    tr_score = reg.score(tr_samples[:, :-1], tr_samples[:, -1])
    te_score = reg.score(te_samples[:, :-1], te_samples[:, -1])
    print(tr_score, te_score)
    return te_score

def modality_cla(exp_name):
    result = []
    for out_layer in range(0, 7):
        a_feat = np.load('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(
            out_layer) + '/audio_rep_{:s}.npy'.format(str(out_layer)))
        v_feat = np.load('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(
            out_layer) + '/video_rep_{:s}.npy'.format(str(out_layer)))
        result.append(cal_modality_cla(a_feat, v_feat))
    return result

root_path = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'
exp_list = get_immediate_subdirectories(root_path)
exp_list.sort()
result = []
target_exp = 65
for exp_name in exp_list:
    try:
        exp_id = int(exp_name[6:8])
        exp_type = exp_name[4:6]
        if exp_id > target_exp and exp_type == 'fm':
            cla_result = modality_cla(exp_name)
            result.append([exp_name] + cla_result)
            np.savetxt('./modality_cla_summary_' + str(target_exp) + '_all.csv', result, delimiter=',', fmt='%s')
    except Exception as e:
        print(e)
        #print(exp_name + 'not processed')