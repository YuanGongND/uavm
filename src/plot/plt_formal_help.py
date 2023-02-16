# -*- coding: utf-8 -*-
# @Time    : 6/30/22 4:32 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : plt_formal_1.1.py

# generate sequence of filename for plot for test set

import numpy as np
import os
os.environ['MPLCONFIGDIR'] = './matcache/'
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from typing import Union,List
import numpy
import cv2
import os
import json
import string

dataset_json_file = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/preprocess/datafiles/test_vgg_convnext_2.json'
label_set = '/data/sls/scratch/yuangong/avbyol/egs/vggsound/preprocess/class_labels_indices_vgg.csv'
label_set = np.loadtxt(label_set, delimiter=',', skiprows=1, usecols=(2), dtype=str)

video_list = []
audio_list = []
label_list = []
with open(dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
    data = data_json['data']
    print('before filter {:d} files'.format(len(data)))
    for entry in data:
        video = entry['video']
        audio = entry['wav']
        label = entry['labels']
        label = int(label.split('_')[-1])
        label = label_set[label].replace("_", ";")
        label = string.capwords(label)
        label = '\_'.join(label.split(' '))
        video_list.append(video)
        audio_list.append(audio)
        label_list.append(label)

np.savetxt('vgg_test_audio_list.csv', audio_list, delimiter=',', fmt='%s')
np.savetxt('vgg_test_video_list.csv', video_list, delimiter=',', fmt='%s')
np.savetxt('vgg_test_label_list.csv', label_list, delimiter=',', fmt='%s')