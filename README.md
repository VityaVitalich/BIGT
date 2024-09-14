# BIGT

This repository contains data for "Do I look like a “cat.n.01” to you? Taxonomy Image Generation Benchmark" and code to reproduce the results in our paper.

## Data

The datasets are located in ```./data``` directory. 

The dataset named ```main_reproducing``` already contains labels for GPT4 and Human Evaluation, as well as pairs of models, that had battle for this lemma. 
This might be helpful for researchers to make observations and fine-tune preference models with off-policy setting. 

The dataset named ```main_bench``` contain exactly same data, however does not include precomputed preferences, assigned models for battles and no labels from human assessors. 
This might be helpful for researchers, that would like to evaluate their model or set of models from scratch

The generated images are availible in HuggingFace repo.

## Code

Our code contains scripts for
1) evaluating ELO Scores
2) evaluating CLIP-Scores needed to calculate metrics from paper
3) evaluation of reward model (however it needs installation from source https://github.com/THUDM/ImageReward)
4) generation of images with models listed in paper
5) evaluation of FID and IS, as described in paper


