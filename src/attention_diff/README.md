## Attention Map Analysis (Figure 5 of the UAVM Paper)

This is to show the interesting results of the audio/visual attention map and audio-visual attention difference (Figure 5 of the UAVM paper).

1.Generate the attention map of models using `gen_att_map_vggsound_summary.py` and `gen_att_map_vggsound_full_att_summary.py` for unified model and modality-independent model, and cross-modality attention model, respectively. `gen_att_map_vggsound_summary.py` also calculates the difference between audio and visual attention maps 

2.Plot the attention maps of unified model, modality-independent model, and cross-modality attention model using `plt_attmap_unified.py` and `plt_attmap_baseline.py`.

Note: you can of course train your own model using our provide scripts and do analysis based on the models. But for the purpose of easier reproduction and analysis without re-training the models, we also release pretrained model and attention maps [[**here**]](https://www.dropbox.com/sh/l2dkmdgc30mkjgm/AACvzmQQo2v7P0iejiRpROG9a?dl=0).