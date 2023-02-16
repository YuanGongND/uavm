## Audio-Visual Embedding Analysis (Figure 3 of the UAVM Paper)

This is to reproduce Figure 3 of the UAVM Paper which analyzes the embedding of the unified models.

1. Generate the intermediate representations of a model using `gen_intermediate_representation.py` (this script does more than that, but you can ignore other functions, we have also released these intermediate representations if you don't want to do your self, please see below).
2. Build a modality classifier and record the results using `check_modality_classification_summary.py`, which will generate a `modality_cla_summary_65_all.csv` (`65` is just internal experiment id, you can ignore it, we have include this file in the repo).
3. Plot the modality classification results using `plt_figure_2_abc.py` (Figure 2 (a), (b), and (c) of the UAVM paper).
4. Plot the t-SNE results using `plt_figure_2_d.py` (Figure 2 (d) of the UAVM paper).

Note: you can of course train your own model using our provide scripts and do analysis based on the models. But for the purpose of easier reproduction and analysis without re-training the models, we also release all models and corresponding intermediate representations of the models [[**here**]](https://www.dropbox.com/sh/np9fydo2q6yabj7/AAAKzZHq3Q4_ckGV_ohaqtj3a?dl=0), it is very large at around 30GB.