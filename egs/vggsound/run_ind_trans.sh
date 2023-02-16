#!/bin/bash
##SBATCH -p sm,1080
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-2,sls-sm-[1,5,6]
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="vgg_ind"
#SBATCH --output=../log/%j_vgg_ind.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../venv/bin/activate
export TORCH_HOME=../../pretrained_models

s_embed_dim=1024 # high-layer embedding dimension, this is to make fair comparison with UAVM
depth_i=3
depth_s=3
lr=5e-5
r=1 # repeat experiments id

a_tr_prob=0.5
input_dim=1024
embed_dim=1024
num_heads=4
mixup=0.5

model=ind_trans
dataset=vggsound
noise=True
bal=bal
epoch=10
lrscheduler_start=1
lrscheduler_decay=0.5
lrscheduler_step=1
lr_adapt=False
tr_data=./datafile/train_vgg_convnext_2.json
te_data=./datafile/test_vgg_convnext_2.json
batch_size=144
label_smooth=0.1

a_seq_len=30
v_seq_len=30
feat_norm=True

exp_dir=./exp/test01-vgg-${model}-lr${lr}-bs-${batch_size}-bal${bal}-mix${mixup}-ls${lrscheduler_start}-ld${lrscheduler_decay}-lst${lrscheduler_step}-lda${lr_adapt}-e${embed_dim}-se${s_embed_dim}-h${num_heads}-di${depth_i}-ds${depth_s}-asl${a_seq_len}-vsl${v_seq_len}-norm${feat_norm}-a${a_tr_prob}-noise${noise}-r${r}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./datafile/class_labels_indices_vgg.csv --n_class 309 \
--loss CE --metrics acc \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--mixup ${mixup} --bal ${bal} --label_smooth ${label_smooth} --noise ${noise} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} --lr_adapt ${lr_adapt} \
--embed_dim ${embed_dim} --s_embed_dim ${s_embed_dim} --num_heads ${num_heads} --depth_i ${depth_i} --depth_s ${depth_s} \
--input_dim ${input_dim} --a_seq_len ${a_seq_len} --v_seq_len ${v_seq_len} --feat_norm ${feat_norm} --a_tr_prob ${a_tr_prob}