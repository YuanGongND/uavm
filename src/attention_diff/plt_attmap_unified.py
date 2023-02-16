# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_attmap_unified.py

# visualize attention map of 30*30 model

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import numpy
import numpy as np
from matplotlib import pyplot as plt

root_path = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/'
embed_dim = 1024

exp_name_list = ['testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se' + str(embed_dim) + '-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1']

exp = exp_name_list[0]

audio_attmap = np.load(root_path + exp + '/attmap/audio_attmap_5.npy')
video_attmap = np.load(root_path + exp + '/attmap/video_attmap_5.npy')
print(audio_attmap.shape, video_attmap.shape)


for max_idx in range(1):
    max_audio_attmap = audio_attmap[max_idx]
    max_video_attmap = video_attmap[max_idx]
    plt.subplot(1,2,1)
    head_max = max_audio_attmap.max(axis=2)
    max_audio_attmap = max_audio_attmap / head_max[:, :, np.newaxis]
    plt.imshow(np.sum(max_audio_attmap, axis=1).reshape(4, 30))
    plt.xticks([])
    plt.xlabel('Time')
    plt.yticks([0, 1, 2, 3], ['h1', 'h2', 'h3', 'h4'])
    plt.subplot(1,2,2)
    head_max = max_video_attmap.max(axis=2)
    max_video_attmap = max_video_attmap / head_max[:, :, np.newaxis]
    plt.imshow(np.sum(max_video_attmap, axis=1).reshape(4, 30))
    plt.xticks([])
    plt.xlabel('Time')
    plt.yticks([0,1,2,3], ['h1','h2','h3','h4'])

figure = plt.gcf()
figure.set_size_inches(8, 20)
plt.savefig('./att_map.png')