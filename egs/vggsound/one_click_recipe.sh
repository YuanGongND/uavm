#!/bin/bash       
#title           :one_click_recipe.sh
#description     :
#author		     :Yuan Gong @ MIT
#time            :12/3/22 4:53 PM
#=======================================

mkdir ./vgg_feat

# run everything in one click (download our pretrained feature)

# download pretrained audio and visual features and unzip
# if unzip is not installed, install it with "sudo apt-get install unzip"
wget https://www.dropbox.com/s/qv1umuxg1rq4dzg/v_feat.zip?dl=1 -O ./vgg_feat/v_feat.zip
unzip ./vgg_feat/v_feat.zip -d ./vgg_feat/

wget https://www.dropbox.com/s/px08p7zgovh7b3p/a_feat.zip?dl=0 -O ./vgg_feat/a_feat.zip
unzip ./vgg_feat/a_feat.zip -d ./vgg_feat/

# remove the zip file to save space
rm ./vgg_feat/v_feat.zip
rm ./vgg_feat/a_feat.zip

# download sample json files
wget https://www.dropbox.com/s/mnx6e7vt3fyx22p/train_vgg_convnext_2.json?dl=1 -O ./datafile/train_vgg_convnext.json
wget https://www.dropbox.com/s/9k7ljjm0hf21wjm/test_vgg_convnext_2.json?dl=1 -O ./datafile/test_vgg_convnext.json

# adapt local path to the json files
python ./datafile/adapt_json_datafiles.py

# create sample weight file (for class balanced sampling)
python ./datafile/gen_weight_file.py --label_indices_path ./datafile/class_labels_indices_vgg.csv --datafile_path ./datafile/train_vgg_convnext.json

# start to train the uavm model
set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../venv/bin/activate
export TORCH_HOME=../../pretrained_models

s_embed_dim=1024
depth_i=3
depth_s=3
lr=5e-5
a_tr_prob=0.5
r=3 # repeat experiments id

input_dim=1024
embed_dim=1024
num_heads=4
mixup=0.5

model=uavm
dataset=vggsound
noise=True
bal=bal
epoch=10
lrscheduler_start=1
lrscheduler_decay=0.5
lrscheduler_step=1
lr_adapt=False
tr_data=./datafile/train_vgg_convnext.json
te_data=./datafile/test_vgg_convnext.json
batch_size=144
label_smooth=0.1

a_seq_len=30
v_seq_len=30
feat_norm=True

exp_dir=./exp/test02-vgg-${model}-lr${lr}-bs-${batch_size}-bal${bal}-mix${mixup}-ls${lrscheduler_start}-ld${lrscheduler_decay}-lst${lrscheduler_step}-lda${lr_adapt}-e${embed_dim}-se${s_embed_dim}-h${num_heads}-di${depth_i}-ds${depth_s}-asl${a_seq_len}-vsl${v_seq_len}-norm${feat_norm}-a${a_tr_prob}-noise${noise}-r${r}
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
