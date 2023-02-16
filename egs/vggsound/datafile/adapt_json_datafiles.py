# -*- coding: utf-8 -*-
# @Time    : 12/3/22 5:05 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : adapt_json_datafiles.py

# create sample json file

import os
import json

def adapt_json_train(dataset_json_file='./datafile/train_vgg_convnext.json'):
    cur_path = os. getcwd()
    new_data = []
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
        data = data_json['data']
        print('before adapt {:d} files'.format(len(data)))
        for entry in data:
            labels = entry['labels']
            video_id = entry['afeat'].split('/')[-1][:-4]
            afeat = cur_path + '/vgg_feat/audio_feat_convnext_2/' + video_id + '.npy'
            vfeat = cur_path + '/vgg_feat/video_feat_convnext_2/' + video_id + '.npy'
            new_entry = {}
            new_entry['labels'] = labels
            new_entry['afeat'] = afeat
            new_entry['vfeat'] = vfeat
            new_data.append(new_entry)

    output = {'data': new_data}
    print('after adapt {:d} files'.format(len(new_data)))
    with open('./datafile/train_vgg_convnext.json', 'w') as f:
        json.dump(output, f, indent=1)

def adapt_json_test(dataset_json_file='./datafile/test_vgg_convnext.json'):
    cur_path = os. getcwd()
    new_data = []
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
        data = data_json['data']
        print('before adapt {:d} files'.format(len(data)))
        for entry in data:
            labels = entry['labels']
            video_id = entry['afeat'].split('/')[-1][:-4]
            afeat = cur_path + '/vgg_feat/audio_feat_convnext_2/' + video_id + '.npy'
            vfeat = cur_path + '/vgg_feat/video_feat_convnext_2/' + video_id + '.npy'
            new_entry = {}
            new_entry['labels'] = labels
            new_entry['afeat'] = afeat
            new_entry['vfeat'] = vfeat
            new_data.append(new_entry)

    output = {'data': new_data}
    print('after adapt {:d} files'.format(len(new_data)))
    with open('./datafile/test_vgg_convnext.json', 'w') as f:
        json.dump(output, f, indent=1)

adapt_json_train()
adapt_json_test()