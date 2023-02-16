# -*- coding: utf-8 -*-
# @Time    : 12/2/22 4:08 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : copy_feats.py

# copy audio/visual intermediate representations and corresponding models from experimental path to formal release path

import os

exp_name_list = [
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se32-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se128-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se512-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se16-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se64-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm69-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se256-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue',
'testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue']

exp_list = []
for exp_name in exp_name_list:
    for r in ['-r1', '-r2', '-r3']:
        exp_list.append(exp_name + r)
print(exp_list)
result = []

for exp_name in exp_list:
    #os.system('cp -r /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/feat' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/embedding/' + exp_name)
    os.system('cp /data/sls/scratch/yuangong/avbyol/egs/vggsound/exp/' + exp_name + '/models/best_audio_model.pth' + ' /data/sls/scratch/yuangong/uavm/pretrained_models/embedding/' + exp_name + '/best_audio_model.pth')