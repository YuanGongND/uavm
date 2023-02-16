#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2,3],sls-sm-[5,6]
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="ast_vgg_cnn"
#SBATCH --output=../log/%j_as_cnn.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../venv/bin/activate
export TORCH_HOME=../../pretrained_models

model=convnext_ori
model_size=2
dataset=audioset
dataset_mean=-5.081
dataset_std=4.4849
target_length=1000
noise=True
# full or balanced for audioset
imagenetpretrain=True

bal=bal
lr=1e-4
epoch=25
lrscheduler_start=15
lrscheduler_decay=0.5
lrscheduler_step=1
wa=False
wa_start=15
wa_end=20
lr_adapt=False
tr_data=./datafile/train_vgg_convnext_2.json
te_data=./datafile/test_vgg_convnext_2.json
freqm=48
timem=192
mixup=0.5
batch_size=120
label_smooth=0.1

exp_dir=./exp/test01-${set}-${model}-${model_size}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-lda${lr_adapt}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_feat.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./datafile/class_labels_indices_vgg.csv --n_class 309 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--imagenet_pretrain $imagenetpretrain --model_size ${model_size} --label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics acc --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt}