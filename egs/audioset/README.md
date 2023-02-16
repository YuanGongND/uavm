## AudioSet Recipe

The AudioSet recipe is almost identical with the VGGSound recipe, expect that we cannot provide pretrained features. 

0. Download the AudioSet dataset, create json datafiles to include the wav path, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
1. Train an audio feature extractor with the AudioSet audio data using `run_feat_extractor.sh`, which will call `src/run_feat.py`, which will call `src/traintest_feat.py`. (You can skip step 1&2 by using our pretrained audio feature extractor [[**here**](https://www.dropbox.com/s/ld2q0bsyiwzia86/as_audio_feat.pth?dl=1)])
2. Generate audio features and save them on disk using `src/gen_audio_feat/gen_audio_feature.py`. (For full AudioSet-2M, this process needs to be parallelized so that it can be finished in reasonable time).
3. Generate visual features and save them on disk using scripts in `src/gen_video_feat`. (For full AudioSet-2M, this process needs to be parallelized so that it can be finished in reasonable time).
4. Create json datafiles to include the path of audio/visual feature path and label information, see samples at [[**here**]](https://www.dropbox.com/sh/pvvs3vd6mx3sbah/AADOroMNU_Sb2r67_CzBKw1Aa?dl=1), generate sample weight file using `./datafile/gen_weight_file`.
5. Train the uavm, cross-modality attention model, and modality-independent model using `run_uavm.sh`, `run_fullatt_trans.sh`, and `run_ind_trans.sh`, respectively. 

