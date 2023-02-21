## Recipes to Reproduce 65.8\% Accuracy on VGGSound

### Option 1 (Recommended). One-Click Recipe
We recommend to start with `one_click_recipe.sh` that loads our pretrained features, do all preprocessing, and train the UAVM model first. It should report a SOTA accuracy on VGGSound.

### Option 2. Step-by-Step Recipe to Do Everything from Scratch
Alternatively, you can also start from scratch. We provide everything you would need to do so. Note: You can skip step 0-4 and directly go to step 5 if you don't want to download the dataset by yourself and train your own feature extractor.

0. Download the [VGGSound dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), create json datafiles to include the wav path, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
1. Train an audio feature extractor with the VGGSound audio data using `run_feat_extractor.sh`, which will call `src/run_feat.py`, which will call `src/traintest_feat.py`. (You can skip this step by using our pretrained feature extractor from [[**this link**]](https://www.dropbox.com/s/aduqm8m4gpjaygh/vgg_audio_feat.pth?dl=1)).
2. Generate audio features and save them on disk using `src/gen_audio_feat/gen_audio_feature.py`. (You can skip step 1&2 by using our pretrained audio features from [[**this link**]](https://www.dropbox.com/s/px08p7zgovh7b3p/a_feat.zip?dl=1)).
3. Generate visual features and save them on disk using scripts in `src/gen_video_feat`. (You can skip this step by using our pretrained visual features from [[**this link**]](https://www.dropbox.com/s/qv1umuxg1rq4dzg/v_feat.zip?dl=1)).
4. Create json datafiles to include the path of audio/visual feature path and label information, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
5. Train the uavm, cross-modality attention model, and modality-independent model using `run_uavm.sh`, `run_fullatt_trans.sh`, and `run_ind_trans.sh`, respectively. 

