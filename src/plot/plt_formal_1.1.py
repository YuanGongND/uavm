# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_formal_1.1.py

# plot the performance of 1) full att; 2) unified trans with 3-3; 3) indepedent for different shared dimension.

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def get_mean_std(exp_name):
    root_path = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'
    three_res = []
    for repeat in ['-r1', '-r2', '-r3']:
        #print(repeat)
        cur_res = np.loadtxt(root_path + exp_name + repeat + '/result.csv', delimiter=',') * 100
        # print(cur_res[:, 6])
        # print(np.argmax(cur_res[:, 6]))
        # print(cur_res.shape)
        three_res.append(cur_res)
    three_res = np.stack(three_res)
    res_mean = np.mean(three_res, axis=0)
    res_std = np.std(three_res, axis=0)
    #max_idx = np.argmax(res_mean[:, 6])
    max_idx = 24
    res_mean = res_mean[max_idx, [0, 3, 6]]
    res_std = res_std[max_idx, [0, 3, 6]]
    return res_mean[0], res_mean[1], res_mean[2], res_std[0], res_std[1], res_std[2]

# 0=audio, 1=video, 2=av
modality_name = ['Audio', 'Visual', 'Audio-Visual Fusion']
for modality in range(0,3):
    plt.subplot(1,3,modality+1)

    # unified model
    exp_list = \
        ['testfm69-vgg-unified_trans-lr5e-4-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se32-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se512-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

    all_res = []
    for exp in exp_list:
        res = get_mean_std(exp)
        all_res.append(res)

    exp_num = len(exp_list)
    plt.plot([16, 32, 64, 128, 256, 512, 1024], [x[modality] for x in all_res], 's-', label='Unified', color='r', markersize=10)
    plt.fill_between([16, 32, 64, 128, 256, 512, 1024], [x[modality] - x[modality + 3] for x in all_res], [x[modality] + x[modality + 3] for x in all_res], alpha=0.3, color='r')

    # seperate model
    exp_list = [
    'testfm76-vgg-seperate_trans-lr5e-4-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
    'testfm76-vgg-seperate_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se32-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
    'testfm76-vgg-seperate_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
    'testfm76-vgg-seperate_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
    'testfm76-vgg-seperate_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
    'testfm76-vgg-seperate_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se512-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
    'testfm76-vgg-seperate_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue'
    ]

    all_res = []
    for exp in exp_list:
        res = get_mean_std(exp)
        all_res.append(res)

    exp_num = len(exp_list)
    plt.plot([16, 32, 64, 128, 256, 512, 1024], [x[modality] for x in all_res], 'o-', label='Independent', color='b', markersize=10)
    plt.fill_between([16, 32, 64, 128, 256, 512, 1024], [x[modality] - x[modality + 3] for x in all_res], [x[modality] + x[modality + 3] for x in all_res], alpha=0.3, color='b')

    if modality == 2:
        # full attention model
        exp_list = [
        'testfm75-vgg-full_att_trans-lr5e-4-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se32-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se512-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
        'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

        all_res = []
        for exp in exp_list:
            res = get_mean_std(exp)
            all_res.append(res)

        exp_num = len(exp_list)
        plt.plot([16, 32, 64, 128, 256, 512, 1024], [x[modality] for x in all_res], '^-', label='Cross-Modal\nAttention', color='g', markersize=10)
        plt.fill_between([16, 32, 64, 128, 256, 512, 1024], [x[modality]-x[modality+3] for x in all_res], [x[modality]+x[modality+3] for x in all_res], alpha =0.3, color='g')

    plt.xscale('log')
    plt.xticks([16,64,256,1024], [16,64,256,1024], fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(modality_name[modality], fontsize=16)
    plt.grid()

    if modality == 0:
        plt.ylabel('Accuracy', fontsize=16)
        plt.legend(loc = 'lower right')
        #plt.ylim([0.54, 0.575])
        plt.yticks([55, 56], [55, 56], fontsize=12)
        plt.legend(fontsize=12)
    if modality == 1:
        plt.xlabel('Shared Embedding Dimension ($\it{S_{dim}}$)', fontsize=16)
        plt.legend(fontsize=12)
    if modality == 2:
        plt.yticks([58, 60, 62, 64], [58, 60, 62, 64], fontsize=12)
        plt.legend(fontsize=11)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

figure = plt.gcf()
figure.set_size_inches(9, 2.5)
# plt.xlabel('Shared Embedding Dim')
# plt.ylabel('Accuracy')
#figure.supxlabel('Shared Embedding Dimension')
# figure.supylabel()
plt.savefig('performance_dimension.pdf', bbox_inches='tight', dpi=300)
    #plt.close()