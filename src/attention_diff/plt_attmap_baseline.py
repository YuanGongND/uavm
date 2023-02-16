# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_attmap_baseline.py

# visualize attention map of 30*30 model

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import cv2
import string
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

label_list = np.loadtxt('/data/sls/scratch/yuangong/avbyol/src/plot/vgg_test_label_list.csv', delimiter=',', dtype=str)

root_path = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'
fontsize=8
for max_idx in range(200):

    exp = 'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1'
    exp_name_list = ['testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di6-ds0-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1',
                     'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1',
                     'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1']
    exp_name_tag_list = ['Independent Model, Dim=1024', 'Unified Model, Dim=16', 'Unified Model, Dim=1024']

    # for cross attention models
    av_attmap = np.load(root_path + exp + '/attmap/av_attmap_2.npy')
    max_audio_attmap =av_attmap[max_idx]
    plt.subplot(len(exp_name_list) + 2, 1, 1)
    head_max = max_audio_attmap.max(axis=2)
    max_audio_attmap = max_audio_attmap / head_max[:, :, np.newaxis]
    plt.imshow(np.sum(max_audio_attmap, axis=1).reshape(4, 60))
    plt.xticks([])
    plt.yticks([0, 1, 2, 3], ['h1', 'h2', 'h3', 'h4'], fontsize=6)
    plt.title(r"$\bfLabel: $" + r"$\bf" + label_list[max_idx] + "$" + '\nCross-Modal Attention Model', fontsize=fontsize)

    # for seperate modality models
    for i, exp in enumerate(exp_name_list):
        audio_attmap = np.load(root_path + exp + '/attmap/audio_attmap_5.npy')
        video_attmap = np.load(root_path + exp + '/attmap/video_attmap_5.npy')
        max_audio_attmap = audio_attmap[max_idx]
        max_video_attmap = video_attmap[max_idx]
        plt.subplot(len(exp_name_list)+2, 2, 2*i+3)
        head_max = max_audio_attmap.max(axis=2)
        max_audio_attmap = max_audio_attmap / head_max[:, :, np.newaxis]
        plt.imshow(np.sum(max_audio_attmap, axis=1).reshape(4, 30))
        plt.title(exp_name_tag_list[i] + ' (Audio)', fontsize=fontsize)
        plt.xticks([])
        plt.yticks([0, 1, 2, 3], ['h1', 'h2', 'h3', 'h4'], fontsize=6)
        plt.subplot(len(exp_name_list)+2, 2, 2*i+4)
        head_max = max_video_attmap.max(axis=2)
        max_video_attmap = max_video_attmap / head_max[:, :, np.newaxis]
        plt.imshow(np.sum(max_video_attmap, axis=1).reshape(4, 30))
        plt.xticks([])
        plt.yticks([0, 1, 2, 3], ['h1', 'h2', 'h3', 'h4'], fontsize=6)
        plt.title(exp_name_tag_list[i] + ' (Video)', fontsize=fontsize)

    plt.subplot(len(exp_name_list)+2, 2, 2*i+5)
    img = np.array(Image.open('./spec/{:d}.png'.format(max_idx)))
    plt.imshow(img)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.show()
    plt.title('Audio Spectrogram', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.subplot(len(exp_name_list)+2, 2, 2*i+6)
    img = np.array(Image.open('./vframe/{:d}.png'.format(max_idx)))
    plt.imshow(img)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.title('Video Frames', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)

    figure = plt.gcf()
    figure.set_size_inches(6, 3.5)
    plt.savefig('./att_map/{:d}.png'.format(max_idx), bbox_inches='tight', dpi=500)
    plt.close()
