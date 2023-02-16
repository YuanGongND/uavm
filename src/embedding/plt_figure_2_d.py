# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_formal_1.1.py

# domain invariance representation, tsne

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm

def get_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

# get mean
def get_batch_similarity(a, b):
    B = a.shape[0]
    similarity = []
    for i in range(B):
        cur_sim = get_similarity(a[i, :], b[i, :])
        similarity.append(cur_sim)
    mean_sim = np.mean(similarity)
    return mean_sim

def get_feat_tsne(exp_name, out_layer, downsample):
    root_path = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'
    audio_feat_list = np.load(root_path + exp_name + '/feat/' + str(out_layer) + '/audio_rep_' + str(out_layer) + '.npy')
    video_feat_list = np.load(root_path + exp_name + '/feat/' + str(out_layer) + '/video_rep_' + str(out_layer) + '.npy')

    audio_feat_list = audio_feat_list[::downsample, :]
    video_feat_list = video_feat_list[::downsample, :]

    similarity = get_batch_similarity(audio_feat_list, video_feat_list)
    print('Output of layer {:d} similarity {:.3f}'.format(out_layer, similarity))

    # audio feat on the top
    x = np.concatenate((audio_feat_list, video_feat_list), axis=0)

    tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=5000, init='pca')
    #tsne = TSNE(n_iter=5000)
    z = tsne.fit_transform(x)
    print(z.shape)
    return z

downsample = 5
show_point = 500
title_list = ['Projected\nInput', 'Layer 1\n(Specific)', 'Layer 2\n(Specific)', '(D)\nLayer 3\n(Specific)', 'Layer 4\n(Shared)', 'Layer 5\n(Shared)', 'Layer 6\n(Shared)']

for out_layer in range(7):
    plt.subplot(2, 7, out_layer+1)
    z = get_feat_tsne('testfm69-vgg-unified_trans-lr5e-4-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1', out_layer, downsample)
    num_samples = int(z.shape[0] / 2)
    plt.scatter(z[0:0+show_point, 0], z[0:0+show_point, 1], c='r', s=1, marker='o', label='Audio')
    plt.scatter(z[num_samples:num_samples+show_point, 0], z[num_samples:num_samples+show_point, 1], c='b', s=1, marker='s', label='Video')
    plt.title(title_list[out_layer], fontsize=17)
    plt.xticks([])
    plt.yticks([])
    if out_layer == 0:
        plt.ylabel('$\it{S_{dim}}$=16', fontsize=16)

for out_layer in range(7):
    plt.subplot(2, 7, out_layer+8)
    z = get_feat_tsne('testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1', out_layer, downsample)
    num_samples = int(z.shape[0] / 2)
    plt.scatter(z[0:0+show_point, 0], z[0:0+show_point, 1], c='r', s=1, marker='o', label='Audio')
    plt.scatter(z[num_samples:num_samples+show_point, 0], z[num_samples:num_samples+show_point, 1], c='b', s=1, marker='s', label='Video')
    plt.xticks([])
    plt.yticks([])
    if out_layer == 0:
        plt.ylabel('$\it{S_{dim}}$=1024', fontsize=16)

figure = plt.gcf()
figure.set_size_inches(12, 3)
plt.savefig('tsne_{:d}.pdf'.format(downsample), bbox_inches='tight', dpi=300)