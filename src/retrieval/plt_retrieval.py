# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_retrieval.py

# plot retrieval performance (Figure 4 of the UAVM paper)

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def load_files(filelist):
    all_res = []
    all_res_name = []
    for summary_file in filelist:
        cur_res = np.loadtxt(summary_file, delimiter=',', usecols=list(range(1, 9)))
        cur_res_name = np.loadtxt(summary_file, delimiter=',', dtype=str, usecols=[0])
        all_res.append(cur_res)
        all_res_name.append(cur_res_name)
    all_res = np.concatenate(all_res, axis=0)
    all_res_name = np.concatenate(all_res_name, axis=0)
    return all_res * 100, all_res_name

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
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

plt.subplot(1,2,1)

res, res_name = load_files(['/data/sls/scratch/yuangong/uavm/src/retrieval/retrieval_summary_final_r5.csv'])
res_list = get_result_batch(exp_name_list, res, res_name)

for layer in [-2]:
    plt.plot([16, 32, 64, 128, 256, 512, 1024], [x[0][layer] for x in res_list], 'o-', color='r', markersize=10, label='R@5')
    plt.fill_between([16, 32, 64, 128, 256, 512, 1024], [x[0][layer]-x[1][layer] for x in res_list], [x[0][layer]+x[1][layer] for x in res_list], alpha=0.3, color='r')

res, res_name = load_files(['/data/sls/scratch/yuangong/uavm/src/retrieval/retrieval_summary_final_r10.csv'])
res_list = get_result_batch(exp_name_list, res, res_name)

for layer in [-2]:
    plt.plot([16, 32, 64, 128, 256, 512, 1024], [x[0][layer] for x in res_list], 's-', color='b', markersize=10, label='R@10')
    plt.fill_between([16, 32, 64, 128, 256, 512, 1024], [x[0][layer]-x[1][layer] for x in res_list], [x[0][layer]+x[1][layer] for x in res_list], alpha=0.3, color='b')

figure = plt.gcf()
plt.grid()
plt.xscale('log')
plt.legend(fontsize=12, loc=10)
#plt.ylim([0.1, 0.55])
plt.xticks([16, 64, 256, 1024], [16, 64, 256, 1024], fontsize=12)
#plt.title('(A)', fontsize=14)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xlabel('Shared Embedding Dimension ($\it{S_{dim}}$)', fontsize=14)
plt.ylabel('Retrieval Performance', fontsize=14)

plt.subplot(1,2,2)

exp_name_list = ['testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di0-ds6-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di6-ds0-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

res, res_name = load_files(['/data/sls/scratch/yuangong/uavm/src/retrieval/retrieval_summary_final_r10.csv'])
res_list = get_result_batch(exp_name_list, res, res_name)

end = 8
plt.plot(range(end), res_list[0][0][0:end], 'o-', color='r', markersize=10, label='$\it{N_{s}}$=6')
plt.fill_between(range(end), res_list[0][0][0:end] - res_list[0][1][0:end], res_list[0][0][0:end] + res_list[0][1][0:end], alpha=0.3, color='r')

plt.plot(range(end), res_list[1][0][0:end], 's-', color='b', markersize=10, label='$\it{N_{s}}$=3')
plt.fill_between(range(end), res_list[1][0][0:end] - res_list[1][1][0:end], res_list[1][0][0:end] + res_list[1][1][0:end], alpha=0.3, color='b')

plt.plot(range(end), res_list[2][0][0:end], '^-', color='g', markersize=10, label='$\it{N_{s}}$=0')
plt.fill_between(range(end), res_list[2][0][0:end] - res_list[2][1][0:end], res_list[2][0][0:end] + res_list[2][1][0:end], alpha=0.3, color='g')

figure = plt.gcf()
plt.grid()
plt.legend(fontsize=12, loc=0)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['Input', '1', '2', '3', '4', '5', '6', 'Logits'], fontsize=12)
plt.xlabel('Embedding of Layer', fontsize=14)
plt.ylabel('R@10', fontsize=14)

figure.set_size_inches(9, 3)
plt.savefig('retrieval_all_rev.pdf', bbox_inches='tight', dpi=300)