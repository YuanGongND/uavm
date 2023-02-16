# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

# run the model on the top of the feature

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader_mm as dataloader
import models
import numpy as np
from traintest import train

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "vggsound"])
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--label_smooth", type=float, default=0.0, help="label smoothing factor")

parser.add_argument("--input_dim", type=int, default=768, help="input audio and visual feature dimension")
parser.add_argument("--embed_dim", type=int, default=768, help="modality-specific transformer layer embedding dimension")
parser.add_argument("--s_embed_dim", type=int, default=0, help="modality-sharing transformer layer embedding dimension, set to 0 to be same with embed_dim")
parser.add_argument("--num_heads", type=int, default=3, help="number of Transformer attention heads")
parser.add_argument("--depth_i", type=int, default=1, help="number of modality-specific layers")
parser.add_argument("--depth_s", type=int, default=1, help="number of modality-sharing layers")
parser.add_argument("--a_seq_len", type=int, default=None, help="sequence length of the audio feature")
parser.add_argument("--v_seq_len", type=int, default=None, help="sequence length of the visual feature")

parser.add_argument('--feat_norm', help='if use l2 norm for the resnet video feature', type=ast.literal_eval)
parser.add_argument("--a_tr_prob", type=float, default=0.5, help="modality training weights, the probabilty that audio is being used")

args = parser.parse_args()

audio_conf = {'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'noise':args.noise, 'label_smooth': args.label_smooth, 'feat_norm': args.feat_norm,
              'a_seq_len': args.a_seq_len, 'v_seq_len': args.v_seq_len, 'modality': 'av'}
val_audio_conf = {'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'noise': False, 'label_smooth': 0.0, 'feat_norm': args.feat_norm,
                  'a_seq_len': args.a_seq_len, 'v_seq_len': args.v_seq_len, 'modality': 'av'}

if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.model == 'uavm':
    print('now train a unified audio-visual model (uavm)')
    audio_model = models.UnifiedTrans(embed_dim=args.embed_dim, num_heads=args.num_heads, depth_i=args.depth_i,
                                      depth_s=args.depth_s, label_dim=args.n_class,
                                      a_seq_len=args.a_seq_len, v_seq_len=args.v_seq_len,
                                      a_input_dim=args.input_dim, v_input_dim=args.input_dim, s_embed_dim=args.s_embed_dim)
elif args.model == 'ind_trans':
    print('now train a modality-independent transformer model (baseline)')
    audio_model = models.SeperateTrans(embed_dim=args.embed_dim, num_heads=args.num_heads, depth_i=args.depth_i,
                                      depth_s=args.depth_s, label_dim=args.n_class,
                                      a_seq_len=args.a_seq_len, v_seq_len=args.v_seq_len,
                                      a_input_dim=args.input_dim, v_input_dim=args.input_dim, s_embed_dim=args.s_embed_dim)
elif args.model == 'full_att_trans':
    print('now train a cross-modality attention transformer model (baseline)')
    audio_model = models.FullAttTrans(embed_dim=args.embed_dim, num_heads=args.num_heads, depth_i=args.depth_i,
                                      depth_s=args.depth_s, label_dim=args.n_class,
                                      a_seq_len=args.a_seq_len, v_seq_len=args.v_seq_len,
                                      a_input_dim=args.input_dim, v_input_dim=args.input_dim, s_embed_dim=args.s_embed_dim)
else:
    raise ValueError('model not supported')

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)