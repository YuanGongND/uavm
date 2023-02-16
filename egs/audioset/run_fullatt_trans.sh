#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2],sls-sm-[5,6]
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="as_fullatt"
#SBATCH --output=../log/%j_as_fullatt.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../venv/bin/activate
export TORCH_HOME=../../pretrained_models

s_embed_dim=1024
depth_i=3
depth_s=3
r=1

input_dim=1024
embed_dim=${s_embed_dim}
feature_size=2
num_heads=4
mixup=0.5

model=full_att_trans
dataset=audioset
noise=True
set=full

if [ $set == balanced ]
then
  bal=none
  lr=5e-4
  epoch=50
  lrscheduler_start=35
  lrscheduler_decay=0.5
  lrscheduler_step=5
  lr_adapt=True
  tr_data=/data/sls/scratch/yuangong/avbyol/egs/audioset/preprocess/datafiles/balanced_train_data_type1_2_mean_dave_conv2_formal.json
elif [ $set == full ]
then
  bal=bal
  lr=1e-5
  epoch=10
  lrscheduler_start=1
  lrscheduler_decay=0.5
  lrscheduler_step=1
  lr_adapt=False
  tr_data=/data/sls/scratch/yuangong/avbyol/egs/audioset/preprocess/datafiles/whole_train_data_dave_conv2_formal.json
fi

te_data=/data/sls/scratch/yuangong/avbyol/egs/audioset/preprocess/datafiles/eval_data_dave_conv2_formal.json
batch_size=144
label_smooth=0.1

a_seq_len=30
v_seq_len=30
feat_norm=True

exp_dir=./exp/test01-as_${set}-${model}-lr${lr}-bs-${batch_size}-bal${bal}-mix${mixup}-ls${lrscheduler_start}-ld${lrscheduler_decay}-lst${lrscheduler_step}-lda${lr_adapt}-e${embed_dim}-se${s_embed_dim}-h${num_heads}-di${depth_i}-ds${depth_s}-asl${a_seq_len}-vsl${v_seq_len}-norm${feat_norm}-feat${feature_size}-noise${noise}-r${r}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./datafile/class_labels_indices.csv --n_class 527 \
--loss BCE --metrics mAP \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--mixup ${mixup} --bal ${bal} --label_smooth ${label_smooth} --noise ${noise} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} --lr_adapt ${lr_adapt} \
--embed_dim ${embed_dim} --s_embed_dim ${s_embed_dim} --num_heads ${num_heads} --depth_i ${depth_i} --depth_s ${depth_s} \
--input_dim ${input_dim} --a_seq_len ${a_seq_len} --v_seq_len ${v_seq_len} --feat_norm ${feat_norm}