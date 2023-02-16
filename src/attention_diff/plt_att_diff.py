# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_formal_1.1.py

# attention difference between a-v plots

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as plticker


def load_files(filelist):
    all_res = []
    all_res_name = []
    for summary_file in filelist:
        cur_res = np.loadtxt(summary_file, delimiter=',', usecols=list(range(1, 7)))
        cur_res_name = np.loadtxt(summary_file, delimiter=',', dtype=str, usecols=[0])
        all_res.append(cur_res)
        all_res_name.append(cur_res_name)
    all_res = np.concatenate(all_res, axis=0)
    all_res_name = np.concatenate(all_res_name, axis=0)
    return all_res, all_res_name

res, res_name = load_files(['/data/sls/scratch/yuangong/avbyol/src/visualization/attdiff_summary_all_69_new.csv', '/data/sls/scratch/yuangong/avbyol/src/visualization/attdiff_summary_all_66_new.csv'])

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

fontsize=18
def load_files(filelist):
    all_res = []
    all_res_name = []
    for summary_file in filelist:
        cur_res = np.loadtxt(summary_file, delimiter=',', usecols=list(range(1, 7)))
        cur_res_name = np.loadtxt(summary_file, delimiter=',', dtype=str, usecols=[0])
        all_res.append(cur_res)
        all_res_name.append(cur_res_name)
    all_res = np.concatenate(all_res, axis=0)
    all_res_name = np.concatenate(all_res_name, axis=0)
    return all_res, all_res_name

res, res_name = load_files(['/data/sls/scratch/yuangong/avbyol/src/visualization/attdiff_summary_all_69_new.csv', '/data/sls/scratch/yuangong/avbyol/src/visualization/attdiff_summary_all_66_new.csv'])

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

exp_name_list = ['testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di6-ds0-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di5-ds1-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di4-ds2-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di2-ds4-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di1-ds5-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
                 'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di0-ds6-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

res_list = get_result_batch(exp_name_list, res, res_name)

print(res_list)

for layer in [-1]:
    plt.plot(range(7), [x[0][layer] for x in res_list], 's-', color='b', markersize=10, linewidth=5)
    plt.fill_between(range(7), [x[0][layer]-x[1][layer] for x in res_list], [x[0][layer]+x[1][layer] for x in res_list], alpha=0.3, color='b')

plt.grid()
plt.xlabel('# Shared Layers ($\it{N_{s}}$)', fontsize=fontsize)
loc = plticker.MultipleLocator(base=0.005) # this locator puts ticks at regular intervals
ax = plt.gca()
ax.yaxis.set_major_locator(loc)
ax.yaxis.get_offset_text().set_fontsize(fontsize)
plt.ylabel('Audio-Visual Attention Difference (MAE)', fontsize=fontsize)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.xticks([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
figure = plt.gcf()
figure.set_size_inches(3, 6)
plt.savefig('att_diff_all_v.pdf', bbox_inches='tight', dpi=300)