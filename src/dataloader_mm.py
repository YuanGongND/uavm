# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_mm.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))

        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)

        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.modality = self.audio_conf.get('modality')

        self.a_seq_len = self.audio_conf.get('a_seq_len')
        print('Sequence length of the dataloader of audio branch is {:d}'.format(self.a_seq_len))

        self.v_seq_len = self.audio_conf.get('v_seq_len')
        print('sequence length of the dataloader of video branch is {:d}'.format(self.v_seq_len))

        self.feat_norm = self.audio_conf.get('feat_norm')
        print('Normalize the feature with l2 : ' + str(self.feat_norm))

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['labels'], data_json[i]['vfeat'], data_json[i]['afeat']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['labels'] = np_data[0]
        datum['vfeat'] = np_data[1]
        datum['afeat'] = np_data[2]
        return datum

    def get_feat(self, filename, filename2=None, seq_len=-1, mix_lambda=0):
        if filename2 == None:
            video_tensor = torch.FloatTensor(np.load(filename))
            if video_tensor.shape[0] < seq_len:
                pad = seq_len - video_tensor.shape[0]
                video_tensor = F.pad(video_tensor, [0, 0, 0, pad])
            elif video_tensor.shape[0] > seq_len:
                video_tensor = video_tensor[:seq_len, :]
            if self.feat_norm == True:
                video_tensor = torch.nn.functional.normalize(video_tensor, dim=1)
            return video_tensor
        else:
            video_tensor1 = torch.FloatTensor(np.load(filename))
            video_tensor2 = torch.FloatTensor(np.load(filename2))

            if video_tensor1.shape[0] < seq_len:
                pad = seq_len - video_tensor1.shape[0]
                video_tensor1 = F.pad(video_tensor1, [0, 0, 0, pad])
            elif video_tensor1.shape[0] > seq_len:
                video_tensor1 = video_tensor1[:seq_len, :]

            if self.feat_norm == True:
                video_tensor1 = torch.nn.functional.normalize(video_tensor1, dim=1)

            if video_tensor2.shape[0] < seq_len:
                pad = seq_len - video_tensor2.shape[0]
                video_tensor2 = F.pad(video_tensor2, [0, 0, 0, pad])
            elif video_tensor2.shape[0] > seq_len:
                video_tensor2 = video_tensor2[:seq_len, :]

            if self.feat_norm == True:
                video_tensor2 = torch.nn.functional.normalize(video_tensor2, dim=1)

            video_tensor = mix_lambda * video_tensor1 + (1 - mix_lambda) * video_tensor2
            return video_tensor

    def __getitem__(self, index):
        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            mix_sample_idx = random.randint(0, self.num_samples-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            mix_lambda = np.random.beta(10, 10)
            if 'a' in self.modality:
                afeat = self.get_feat(datum['afeat'], mix_datum['afeat'], seq_len=self.a_seq_len, mix_lambda=mix_lambda)
            if 'v' in self.modality:
                vfeat = self.get_feat(datum['vfeat'], mix_datum['vfeat'], seq_len=self.v_seq_len, mix_lambda=mix_lambda)

            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
            label_indices = torch.FloatTensor(label_indices)

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            if 'a' in self.modality:
                afeat = self.get_feat(datum['afeat'], None, seq_len=self.a_seq_len, mix_lambda=-1)
            if 'v' in self.modality:
                vfeat = self.get_feat(datum['vfeat'], None, seq_len=self.v_seq_len, mix_lambda=-1)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        if self.noise == True:
            tshift = np.random.randint(-15, 15)
            afeat = torch.roll(afeat, tshift, 0)
            vfeat = torch.roll(vfeat, tshift, 0)

        if 'av' in self.modality:
            return afeat, vfeat, label_indices
        elif 'a' in self.modality:
            return afeat, label_indices
        elif 'v' in self.modality:
            return vfeat, label_indices

    def __len__(self):
        return self.num_samples
