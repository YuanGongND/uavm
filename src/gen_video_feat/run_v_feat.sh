#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2,3],sls-sm-[5,6,13]
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="vfeat"
#SBATCH --output=./log/%j_v_feat.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/avbyol/venv_byol/bin/activate
export TORCH_HOME=../../pretrained_models

fold=$1

python extract_convnext.py --csv=./audioset/input_audioset_unbalanced_convnext_2/${fold}.csv --model_size 2 --type=2d --batch_size=64 --num_decoding_thread=4