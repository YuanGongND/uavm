# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_formal_1.1.py

# domain invariance representation, modality classifier coef

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import sys
import inspect
import argparse
import os
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

res_path = './modality_cla_summary_65_all.csv'
res_name = np.loadtxt(res_path, delimiter=',', dtype=str, usecols=[0])
res = np.loadtxt(res_path, delimiter=',', usecols=list(range(1,8))) * 100

def get_result(exp_name, res, res_name):
    exp_num = res.shape[0]
    three_res = []
    for repeat in ['-r1', '-r2', '-r3']:
        print(exp_name + repeat)
        for i in range(exp_num):
            if res_name[i] == exp_name + repeat:
                three_res.append(res[i])
                break
    three_res = np.stack(three_res)
    res_mean = np.mean(three_res, axis=0)
    res_std = np.std(three_res, axis=0)
    return res_mean, res_std

def get_result_batch(exp_name_list, res, res_name):
    res_list = []
    for exp_name in exp_name_list:
        cur_res_mean, cur_res_std = get_result(exp_name, res, res_name)
        res_list.append([cur_res_mean, cur_res_std])
    return res_list

exp_name_list = ['testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

res_list = get_result_batch(exp_name_list, res, res_name)
print(res_list)

plt.subplot(1, 3, 2)

plt.plot(range(7), res_list[0][0], 'o-', color='b', markersize=10, label='$\it{Sdim}$=16')
plt.fill_between(range(7), res_list[0][0] - res_list[0][1], res_list[0][0] + res_list[0][1], alpha=0.3, color='b')

plt.plot(range(7), res_list[1][0], 's-', color='r', markersize=10, label='$\it{Sdim}$=64')
plt.fill_between(range(7), res_list[1][0] - res_list[1][1], res_list[1][0] + res_list[1][1], alpha=0.3, color='r')

plt.plot(range(7), res_list[2][0], '^-', color='y', markersize=10, label='$\it{Sdim}$=256')
plt.fill_between(range(7), res_list[2][0] - res_list[2][1], res_list[2][0] + res_list[2][1], alpha=0.3, color='y')

plt.plot(range(7), res_list[3][0], 'D-', color='g', markersize=10, label='$\it{Sdim}$=1024')
plt.fill_between(range(7), res_list[3][0] - res_list[3][1], res_list[3][0] + res_list[3][1], alpha=0.3, color='g')

plt.grid()
plt.legend(fontsize=13)
plt.title('(B)', fontsize=18)
plt.xlabel('Embedding of Layer', fontsize=18)
#plt.ylabel('Modality Classification Accuracy', fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Input', 1, 2, 3, 4, 5, 6], fontsize=16)
plt.yticks(fontsize=14)

# first plot
res_path = './modality_cla_summary_65_all.csv'
res_name = np.loadtxt(res_path, delimiter=',', dtype=str, usecols=[0])
res = np.loadtxt(res_path, delimiter=',', usecols=list(range(1,8))) * 100

def get_result(exp_name, res, res_name):
    exp_num = res.shape[0]
    three_res = []
    for repeat in ['-r1', '-r2', '-r3']:
        print(exp_name + repeat)
        for i in range(exp_num):
            if res_name[i] == exp_name + repeat:
                three_res.append(res[i])
                break
    three_res = np.stack(three_res)
    res_mean = np.mean(three_res, axis=0)
    res_std = np.std(three_res, axis=0)
    return res_mean, res_std

def get_result_batch(exp_name_list, res, res_name):
    res_list = []
    for exp_name in exp_name_list:
        cur_res_mean, cur_res_std = get_result(exp_name, res, res_name)
        res_list.append([cur_res_mean, cur_res_std])
    return res_list

exp_name_list = ['testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se32-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se512-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',]

res_list = get_result_batch(exp_name_list, res, res_name)

plt.subplot(1, 3, 1)

plt.plot([16, 32, 64, 128, 256, 512, 1024], [x[0][-1] for x in res_list], 's-', color='b', markersize=10)
plt.fill_between([16, 32, 64, 128, 256, 512, 1024], [x[0][-1]-x[1][-1] for x in res_list], [x[0][-1]+x[1][-1] for x in res_list], alpha=0.3, color='b')

plt.xscale('log')
plt.xticks([16, 64, 256, 1024], [16, 64, 256, 1024], fontsize=16)
plt.yticks(fontsize=14)
plt.grid()
plt.xlabel('$\it{S_{dim}}$', fontsize=18)
plt.ylabel('Modality CLASS ACC.', fontsize=18)
plt.title('(A)', fontsize=18)

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
    return reg.coef_

def modality_cla_coef(exp_name):
    out_layer = 6
    a_feat = np.load('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(out_layer) + '/audio_rep_{:s}.npy'.format(str(out_layer)))
    v_feat = np.load('/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat/' + str(out_layer) + '/video_rep_{:s}.npy'.format(str(out_layer)))
    coef = cal_modality_cla(a_feat, v_feat)
    return coef

plt.subplot(1, 3, 3)
marker = '-'

exp_name = 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1'
coef = modality_cla_coef(exp_name)
coef = np.abs(coef)
coef = np.sort(coef)[0,:]
print(coef.shape)
plt.plot(range(coef.shape[0]), np.sort(coef), marker, color='b', label='$\it{Sdim}$=16', linewidth=4)

exp_name = 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1'
coef = modality_cla_coef(exp_name)
coef = np.abs(coef)
coef = np.sort(coef)[0,:]
print(coef.shape)
plt.plot(range(coef.shape[0]), np.sort(coef), marker, color='r', label = '$\it{Sdim}$=64', linewidth=4)

exp_name = 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r2'
coef = modality_cla_coef(exp_name)
coef = np.abs(coef)
coef = np.sort(coef)[0,:]
print(coef.shape)
plt.plot(range(coef.shape[0]), np.sort(coef), marker, color='y', label = '$\it{Sdim}$=256', linewidth=4)

#exp_name = 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1'
exp_name = 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1'
coef = modality_cla_coef(exp_name)
coef = np.abs(coef)
coef = np.sort(coef)[0,:]
print(coef.shape)
plt.plot(range(coef.shape[0]), np.sort(coef), marker, color='g', label = '$\it{Sdim}$=1024', linewidth=4)

plt.grid()
plt.title('(C)', fontsize=18)
plt.xscale('log')
plt.legend(fontsize=13)
plt.ylim([-0.05, 1])
plt.xticks([1, 16, 64, 256, 1024], [1, 16, 64, 256, 1024], fontsize=16)
plt.yticks([0, 1], [0, 1], fontsize=14)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xlabel('Embedding Index', fontsize=18)
plt.ylabel('Classifier Coefficient', fontsize=18)

figure = plt.gcf()
figure.set_size_inches(12, 3.1)
plt.savefig('modality_cla_all.pdf', dpi=300, bbox_inches='tight')

