
# GraphSVX: Shapley Value Explanations for Graph Neural Networks 

This repository contains the source code for the paper "_GraphSVX: Shapley Value Explanations for Graph Neural Networks_"
by Alexandre Duval and Fragkiskos Malliaros. 

If needed, install the required packages contained in requirements.txt

### To explain a model using GraphSVX
To explain the predictions of a model on a node or graph classification task, run the following command
```
python3 script_explain.py --dataset='DATASET_NAME' --model='MODEL_NAME' --info=True
```
where '_DATASET_NAME_' is the dataset name (e.g Cora, PubMed, syn1, syn2, syn4, syn5, syn6 or Mutagenicity) and 
'_MODEL_NAME_' refers to the model used (e.g GAT or GCN). Note that all synthetic datasets exist and Cora/PubMed are downloaded directly. Only Mutagenicity
requires you to go download it on the Internet on your own. 

Hyperparameters for training are specificied in the Appendix of the paper
and are described in the configs.py file. There are several parameters that you might want to specify: 
    - the indexes of the nodes you would like to explain
    -  the number of samples used in GraphSVX
    - some settings of GraphSVX: feat, coal, g, regu, S, hv, fullempty, hops (see configs file)

### To train a model 
If you would like to train your own model on a chosen dataset, run: 
```
python3 script_train.py --dataset='DATASET_NAME' --model='MODEL_NAME' --save=True
```
Otherwise, all trained models (except for Mutagenicity) already exist and can be used directly. 

### Evaluation 
To follow the evaluation setting described in the paper, you should run the files: 
    - script_eval_noise.py: evaluate GraphSVX on noisy dataset and observe number of 
    noisy features/nodes included in explanations 
    - script_eval_gt.py: evaluate GraphSVX on synthetic datasets with a ground truth. 
All parameters are in the configs.py file, along with a small documentation. 


### The structure of the code is as follows: 
In src: 
    - explainers.py: defines GraphSVX and main baselines
    - data.py: import and process the data 
    - models.py: define GNN models
    - train.py: train GNN models
    - utils.py: stores useful variables
    - eval.py: one of the evaluation of the paper, with real world datasets
    - eval_multiclass.py: explain all classes predictions
    - plots.py: code for nice renderings
    - gengraph.py: generates synthetic datasets

Outside: 
    - results: stores visualisation and evaluation results
    - data: contains some datasets, others will be downloaded automatically
    when launching the training script (Cora, PubMed)
    - models: contains our trained models
    - utils: some useful functions to construct datasets, store them, 
    create plots, train models etc. 
