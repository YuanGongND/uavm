# UAVM: Towards Unifying Audio and Visual Models
  * [Introduction](#introduction)
  * [Citing](#citing)
  * [Getting Started](#getting-started)
    + [Prepare the Environment](#prepare-the-environment)
    + [Where is the code?](#where-is-the-code)
  * [VGGSound Recipe](#vggsound-recipe)
    + [Option 1 (Recommended) One-Click Recipe](#option-1-recommended-one-click-recipe)
    + [Option 2. Step-by-Step Recipe to Do Everything from Scratch](#option-2-step-by-step-recipe-to-do-everything-from-scratch)
  * [AudioSet Recipe](#audioset-recipe)
  * [Audio-Visual Embedding Analysis (Figure 3 of the UAVM Paper)](#audio-visual-embedding-analysis-figure-3-of-the-uavm-paper)
  * [Audio-Visual Retrieval Experiments (Figure 4 of the UAVM paper)](#audio-visual-retrieval-experiments-figure-4-of-the-uavm-paper)
  * [Attention Map Analysis (Figure 5 of the UAVM Paper)](#attention-map-analysis-figure-5-of-the-uavm-paper)
  * [Pretrained Models](#pretrained-models)
    + [Pretrained Audio Feature Extractor](#pretrained-audio-feature-extractor)
    + [Pretrained Video Feature Extractor](#pretrained-video-feature-extractor)
    + [AudioSet Pretrained Model](#audioset-pretrained-model)
    + [VGGSound Pretrained Model](#vggsound-pretrained-model)
  * [Contact](#contact)

## News
This paper will be presented as an oral paper at the ICASSP *Audio for Multimedia and Multimodal Processing* Session at 6/6/2023 10:50:00 (Eastern European Summer Time).
 
## Introduction  

<p align="center"><img src="https://github.com/YuanGongND/uavm/blob/main/uavm.png?raw=true" alt="Illustration of AST." width="600"/></p>

This repository contains the official implementation (in PyTorch) of the models and experiments proposed in the IEEE Signal Processing 2022 paper **UAVM: Towards Unifying Audio and Visual Models** ([Yuan Gong](https://yuangongnd.github.io/), [Alexander H. Liu](https://alexander-h-liu.github.io/), [	
Andrew Rouditchenko](http://people.csail.mit.edu/roudi/), [James Glass](https://www.csail.mit.edu/person/jim-glass); MIT CSAIL). [**[IEEE Xplore]**](https://ieeexplore.ieee.org/document/9964072) [**[arxiv]**](https://arxiv.org/abs/2208.00061)

Conventional audio-visual models have independent audio and video branches. In this work, we *unify* the audio and visual branches by designing a **U**nified **A**udio-**V**isual **M**odel (UAVM). The UAVM achieves a new state-of-the-art audio-visual event classification accuracy of 65.8% on VGGSound. More interestingly, we also find a few intriguing properties of UAVM that the modality-independent counterparts do not have.

Performance-wise, our model and training pipeline achieves good results (SOTA on VGGSound):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uavm-a-unified-model-for-audio-visual/multi-modal-classification-on-vgg-sound)](https://paperswithcode.com/sota/multi-modal-classification-on-vgg-sound?p=uavm-a-unified-model-for-audio-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uavm-a-unified-model-for-audio-visual/audio-classification-on-audioset)](https://paperswithcode.com/sota/audio-classification-on-audioset?p=uavm-a-unified-model-for-audio-visual)

To help better understand the pros and cons of this work, we have attached the anonymous reviews and our responses [**here**](https://github.com/YuanGongND/uavm/tree/main/review). We thank the associate editor and anonymous reviewers' invaluable comments.

## Citing  
Please cite our paper if you find this repository useful. 
```  
@ARTICLE{uavm_gong,
  author={Gong, Yuan and Liu, Alexander H. and Rouditchenko, Andrew and Glass, James},
  journal={IEEE Signal Processing Letters}, 
  title={UAVM: Towards Unifying Audio and Visual Models}, 
  year={2022},
  volume={29},
  pages={2437-2441},
  doi={10.1109/LSP.2022.3224688}}
```

  
## Getting Started  

### Prepare the Environment
Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd uavm/ 
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 
```

### Where is the code?
1. The UAVM model, modality-independent model, and cross-modality attention model scripts are in `src/models/mm_trans.py` (`UnifiedTrans`, `SeperateTrans`, and `FullAttTrans`).
2. The training and test scripts are in `src/run.py` and `src/traintest.py`.
3. The ConvNext audio feature extractor model script is in `src/models/convnext.py`.
4. The ConvNext audio feature extractor model training scripts are in `src/run_feat.py` and `src/traintest_feat.py`.
5. Audio and visual feature extraction scripts are in `src/gen_audio_feat/` and `src/gen_video_feat/`, respectively.
6. The VGGSound recipe is in `egs/vggsound`.
7. The AudioSet recipe is in `egs/audioset`.
8. The probing test and plot scripts for UAVM paper figure 3, 4, and 5 are in `src/{embedding, retrieval, attention_diff}`, respectively.

## VGGSound Recipe

This part of code is in `egs/vggsound/`.

### Option 1 (Recommended) One-Click Recipe
We recommend to start with `one_click_recipe.sh` that loads our pretrained features, do all preprocessing, and train the UAVM model first. It should report a SOTA accuracy on VGGSound.

### Option 2. Step-by-Step Recipe to Do Everything from Scratch
Alternatively, you can also start from scratch. We provide everything you would need to do so. Note: You can skip step 0-4 and directly go to step 5 if you don't want to download the dataset by yourself and train your own feature extractor.

0. Download the [VGGSound dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), create json datafiles to include the wav path, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
1. Train an audio feature extractor with the VGGSound audio data using `run_feat_extractor.sh`, which will call `src/run_feat.py`, which will call `src/traintest_feat.py`. (You can skip this step by using our pretrained feature extractor from [[**this link**]](https://www.dropbox.com/s/aduqm8m4gpjaygh/vgg_audio_feat.pth?dl=1)).
2. Generate audio features and save them on disk using `src/gen_audio_feat/gen_audio_feature.py`. (You can skip step 1&2 by using our pretrained audio features from [[**this link**]](https://www.dropbox.com/s/px08p7zgovh7b3p/a_feat.zip?dl=1)).
3. Generate visual features and save them on disk using scripts in `src/gen_video_feat`. (You can skip this step by using our pretrained visual features from [[**this link**]](https://www.dropbox.com/s/qv1umuxg1rq4dzg/v_feat.zip?dl=1)).
4. Create json datafiles to include the path of audio/visual feature path and label information, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
5. Train the uavm, cross-modality attention model, and modality-independent model using `run_uavm.sh`, `run_fullatt_trans.sh`, and `run_ind_trans.sh`, respectively. 

## AudioSet Recipe

This part of code  is in `egs/audioset/`.

The AudioSet recipe is almost identical with the VGGSound recipe, expect that we cannot provide pretrained features. 

0. Download the AudioSet dataset, create json datafiles to include the wav path, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
1. Train an audio feature extractor with the AudioSet audio data using `run_feat_extractor.sh`, which will call `src/run_feat.py`, which will call `src/traintest_feat.py`. (You can skip step 1&2 by using our pretrained audio feature extractor [[**here**](https://www.dropbox.com/s/ld2q0bsyiwzia86/as_audio_feat.pth?dl=1)])
2. Generate audio features and save them on disk using `src/gen_audio_feat/gen_audio_feature.py`. (For full AudioSet-2M, this process needs to be parallelized so that it can be finished in reasonable time).
3. Generate visual features and save them on disk using scripts in `src/gen_video_feat`. (For full AudioSet-2M, this process needs to be parallelized so that it can be finished in reasonable time).
4. Create json datafiles to include the path of audio/visual feature path and label information, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
5. Train the uavm, cross-modality attention model, and modality-independent model using `run_uavm.sh`, `run_fullatt_trans.sh`, and `run_ind_trans.sh`, respectively. 

## Audio-Visual Embedding Analysis (Figure 3 of the UAVM Paper)

This part of code is in `src/embedding/`.

This is to reproduce Figure 3 of the UAVM Paper which analyzes the embedding of the unified models.

1. Generate the intermediate representations of a model using `gen_intermediate_representation.py` (this script does more than that, but you can ignore other functions, we have also released these intermediate representations if you don't want to do yourself, please see below).
2. Build a modality classifier and record the results using `check_modality_classification_summary.py`, which will generate a `modality_cla_summary_65_all.csv` (`65` is just internal experiment id, you can ignore it, we have include this file in the repo).
3. Plot the modality classification results using `plt_figure_2_abc.py` (Figure 2 (a), (b), and (c) of the UAVM paper).
4. Plot the t-SNE results using `plt_figure_2_d.py` (Figure 2 (d) of the UAVM paper).

Note: you can of course train your own model using our provide scripts and do analysis based on the models. But for the purpose of easier reproduction and analysis without re-training the models, we also release all models and corresponding intermediate representations of the models [[**here**]](https://www.dropbox.com/sh/np9fydo2q6yabj7/AAAKzZHq3Q4_ckGV_ohaqtj3a?dl=0), it is very large at around 30GB.

## Audio-Visual Retrieval Experiments (Figure 4 of the UAVM paper).

This part of code is in `src/retrieval/`.

1.Generate the retrieval performance results R@5 and R@10 using `gen_rep_vggsound_retrieval2_r5_summary.py` and `gen_rep_vggsound_retrieval2_r10_summary.py`, which will generate output `retrieval_summary_final_r5.csv` and `retrieval_summary_final_r10.csv` respectively.

2.Plot the figure using `plt_retrieval.py`, which reads the result from `retrieval_summary_final_r5.csv` and `retrieval_summary_final_r10.csv`.

Note: You can download all pretrained models needed to reproduce this result at [[**here**]](https://www.dropbox.com/sh/2k63js2pqiqtre5/AAAzcExQsYf4H74q8cPJ-TLEa?dl=0).

## Attention Map Analysis (Figure 5 of the UAVM Paper)

This part of code is in `src/attention_diff/`.

This is to show the interesting results of the audio/visual attention map and audio-visual attention difference (Figure 5 of the UAVM paper).

1.Generate the attention map of models using `gen_att_map_vggsound_summary.py` and `gen_att_map_vggsound_full_att_summary.py` for unified model and modality-independent model, and cross-modality attention model, respectively. `gen_att_map_vggsound_summary.py` also calculates the difference between audio and visual attention maps 

2.Plot the attention maps of unified model, modality-independent model, and cross-modality attention model using `plt_attmap_unified.py` and `plt_attmap_baseline.py`.

Note: you can of course train your own model using our provide scripts and do analysis based on the models. But for the purpose of easier reproduction and analysis without re-training the models, we also release pretrained model and attention maps [[**here**]](https://www.dropbox.com/sh/l2dkmdgc30mkjgm/AACvzmQQo2v7P0iejiRpROG9a?dl=0).

## Pretrained Models

### Pretrained Audio Feature Extractor 

---
[**AudioSet-2M pretrained ConvNext-Base**](https://www.dropbox.com/s/ld2q0bsyiwzia86/as_audio_feat.pth?dl=1)

[**VGGSound pretrained ConvNext-Base**](https://www.dropbox.com/s/aduqm8m4gpjaygh/vgg_audio_feat.pth?dl=1)

### Pretrained Video Feature Extractor

---
We use `torchvision` official ImageNet pretrained ConvNext-Base as our visual feature extractor.

### AudioSet Pretrained Model

---
[**UAVM Model**](https://www.dropbox.com/s/1gni463q67yx47z/testfm74-full-unified_trans-lr1e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1.pth?dl=1)

[**Modality-Independent Model**](https://www.dropbox.com/s/f690u566mgo8k2r/testfm74-full-unified_trans-lr1e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di6-ds0-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1.pth?dl=0)

[**Cross-Modality Attention Model**](https://www.dropbox.com/s/wlhul07vvcmu1y3/testfm74-full-full_att_trans-lr1e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue.pth?dl=1)

### VGGSound Pretrained Model

---
[**UAVM Model**](https://www.dropbox.com/s/llvbyfokp4gxxwd/testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1.pth?dl=1)

[**Modality-Independent Model**](https://www.dropbox.com/s/exde913yrqovesp/testfm66-vgg-unified_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaFalse-e1024-se1024-h4-di6-ds0-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1.pth?dl=1)

[**Cross-Modality Attention Model**](https://www.dropbox.com/s/twafscfsk3789gj/testfm75-vgg-full_att_trans-lr5e-5-bs-144-balbal-mix0.5-ls1-ld0.5-lst1-ldaTrue-e1024-se1024-h4-di3-ds3-asl30-vsl30-normTrue-feat2-a0.5-noiseTrue-r1.pth?dl=1)

### VGGSound Pretrained Audio and Video Features

[**Audio Features**](https://www.dropbox.com/s/px08p7zgovh7b3p/a_feat.zip?dl=1) (~20GB)

[**Video Features**](https://www.dropbox.com/s/qv1umuxg1rq4dzg/v_feat.zip?dl=1) (~10GB)

## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.