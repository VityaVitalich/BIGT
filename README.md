# BIGT

This repository contains data for Benchmark for Image Generation in Taxonomy and code to reproduce the results in our paper

## Data

The datasets are located in ```./data``` directory. 

The dataset named ```main_reproducing``` already contains labels for GPT4 and Human Evaluation, as well as pairs of models, that had battle for this lemma. 
This might be helpful for researchers to make observations and fine-tune preference models with off-policy setting. 

The dataset named ```main_bench``` contain exactly same data, however does not include precomputed preferences, assigned models for battles and no labels from human assessors. 
This might be helpful for researchers, that would like to evaluate their model or set of models from scratch


## Code
