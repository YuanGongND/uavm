#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-[5,6]
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="ast_as_cnn"
#SBATCH --output=../log/%j_as_cnn.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../venv/bin/activate
export TORCH_HOME=../../pretrained_models

model=convnext_ori
model_size=2
dataset=audioset
dataset_mean=-4.2677
dataset_std=4.5690
target_length=1000
noise=True
# full or balanced for audioset
set=full
imagenetpretrain=True
if [ $set == balanced ]
then
  bal=none
  lr=1e-4
  epoch=60
  lrscheduler_start=35
  lrscheduler_decay=0.5
  lrscheduler_step=5
  wa=True
  wa_start=41
  wa_end=60
  lr_adapt=True
  tr_data=/data/sls/scratch/yuangong/avbyol/egs/audioset/preprocess/datafiles/balanced_train_data_type1_2_mean_dave.json
elif [ $set == full ]
then
  bal=bal
  lr=5e-4
  epoch=30
  lrscheduler_start=10
  lrscheduler_decay=0.5
  lrscheduler_step=1
  wa=True
  wa_start=16
  wa_end=30
  lr_adapt=False
  tr_data=/data/sls/scratch/yuangong/avbyol/egs/audioset/preprocess/datafiles/whole_train_data_dave_3_mix.json
fi
te_data=/data/sls/scratch/yuangong/avbyol/egs/audioset/preprocess/datafiles/eval_data_dave.json
freqm=48
timem=192
mixup=0.5
batch_size=120
label_smooth=0.1

exp_dir=./exp/test01-${set}-${model}-${model_size}-lr-${lr}-bal${bal}-lda${lr_adapt}-lrs${lrscheduler_start}-lrstep${lrscheduler_step}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_feat.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./datafile/class_labels_indices.csv --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--imagenet_pretrain $imagenetpretrain \
--model_size ${model_size} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss BCE --metrics mAP --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt}