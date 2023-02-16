# -*- coding: utf-8 -*-
# @Time    : 12/2/22 4:08 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : copy_feats.py

# copy models and attention maps from experimental path to formal release path

import os

exp_name_list = [
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di6-ds0-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di5-ds1-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di4-ds2-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di2-ds4-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di1-ds5-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di0-ds6-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

exp_list = []
for exp_name in exp_name_list:
    for r in ['-r1']:
        exp_list.append(exp_name + r)
print(exp_list)
result = []

for exp_name in exp_list:
    try:
        os.mkdir('/data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name)
    except:
        pass
    os.system('cp -r /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap/audio_attmap_5.npy' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name + '/audio_attmap_5.npy')
    os.system('cp -r /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap/video_attmap_5.npy' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name + '/video_attmap_5.npy')
    os.system('cp /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/models/best_audio_model.pth' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name + '/best_audio_model.pth')


# cross-modality attention model.
exp_name_list = [
'testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

exp_list = []
for exp_name in exp_name_list:
    for r in ['-r1']:
        exp_list.append(exp_name + r)
print(exp_list)
result = []

for exp_name in exp_list:
    try:
        os.mkdir('/data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name)
    except:
        pass
    os.system('cp -r /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap/audio_attmap_2.npy' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name + '/audio_attmap_2.npy')
    os.system('cp -r /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/attmap/video_attmap_2.npy' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name + '/video_attmap_2.npy')
    os.system('cp /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/models/best_audio_model.pth' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/attention_diff/' + exp_name + '/best_audio_model.pth')


