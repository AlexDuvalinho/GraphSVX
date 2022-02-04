
# GraphSVX: Shapley Value Explanations for Graph Neural Networks 

This repository contains the source code for the paper [GraphSVX: Shapley Value Explanations for Graph Neural Networks_](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_135.pdf), 
by Alexandre Duval and Fragkiskos Malliaros - accepted at the _European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD) 2021_. 

### Abstract
Graph Neural Networks (GNNs) achieve significant performance for various learning tasks on geometric data due to the incorporation of graph structure into the learning of node representations, which renders their comprehension challenging. In this paper, we first propose a unified framework satisfied by most existing GNN explainers. Then, we introduce GraphSVX, a post hoc local model-agnostic explanation method specifically designed for GNNs. GraphSVX is a decomposition technique that captures the “fair” contribution of each feature and node towards the explained prediction by constructing a surrogate model on a perturbed dataset. It extends to graphs and ultimately provides as explanation the Shapley Values from game theory. Experiments on real-world and synthetic datasets demonstrate that GraphSVX achieves state-of-the-art performance compared to baseline models while presenting core theoretical and human-centric properties.

![Framework Image](/utils/pipeline_figure.png)

### Set up 
If needed, install the required packages contained in [requirements.txt](/requirements.txt) as well as three additional packages 
that need to be installed separately (because of their dependency to pytorch). 
```
pip install -r requirements.txt
pip install torch-sparse torch-scatter torch-cluster
```

### To explain a model using GraphSVX
To explain the predictions of a model on a node or graph classification task, run [script_explain.py](/script_explain.py):
```
python3 script_explain.py --dataset='DATASET_NAME' --model='MODEL_NAME' --info=True
```
where '_DATASET_NAME_' is the dataset name (e.g Cora, PubMed, syn1, syn2, syn4, syn5, syn6 or Mutagenicity) and 
'_MODEL_NAME_' refers to the model used (e.g GAT or GCN). Note that all synthetic datasets exist and Cora/PubMed are downloaded directly. Only Mutagenicity
requires you to go download it on the Internet on your own. 

Hyperparameters for training are specificied in the Appendix of the paper
and are described in the configs.py file. There are several parameters that you might want to specify: 
- the indexes of the nodes you would like to explain
- the number of samples used in GraphSVX
- some settings of GraphSVX such as feat, coal, g, regu, S, hv, fullempty, hops (see configs file)

### To train a model 
If you would like to train your own model on a chosen dataset, run [script_train.py](/script_train.py): 
```
python3 script_train.py --dataset='DATASET_NAME' --model='MODEL_NAME' --save=True
```
Otherwise, all trained models (except for Mutagenicity) already exist and can be used directly. 

### Evaluation 
To follow the evaluation setting described in the paper, you should create a results folder and run the files: 
- [script_eval_gt.py](/script_eval_gt.py): evaluate GraphSVX on synthetic datasets with a ground truth. For instance, run this command to evaluate GraphSVX on the BA-Shapes dataset ('syn1'). 
```
python3 script_eval_gt.py --dataset='syn1' --num_samples=400 --S=1 --coal='SmarterSeparate' --feat='Expectation'
python3 script_eval_gt.py --dataset='syn2' --num_samples=800 --S=1 --coal='SmarterSeparate' --feat='All'
python3 script_eval_gt.py --dataset='syn4' --num_samples=1400 --S=4 --coal='SmarterSeparate' --feat='Expectation' 
python3 script_eval_gt.py --dataset='syn5' --num_samples=1000 --S=4 --coal='SmarterSeparate' --feat=‘Expectation’
python3 script_eval_gt.py --dataset='syn6' --num_samples=200 --S=4 --coal='SmarterSeparate' --feat='Expectation'
```
- [script_eval_noise_node.py](/script_eval_noise_node.py): evaluate GraphSVX on noisy dataset and observe number of noisy nodes included in explanations.
```
python3 script_eval_noise_node.py --dataset=Cora --num_samples=800 --hops=2 --hv='compute_pred' --test_samples=40 --model='GAT' --coal='NewSmarterSeparate' --S=3 --regu=0
```
- [script_eval_noise_feat.py](/script_eval_noise_feat.py): evaluate GraphSVX on noisy dataset and observe number of noisy features included in explanations.
```
python3 script_eval_noise_feat.py --dataset=Cora --model=GAT --num_samples=3000 --test_samples=40 --hops=2 --hv=compute_pred_subgraph
```
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
- data: contains some datasets, others will be downloaded automatically when launching the training script (Cora, PubMed)
- models: contains our trained models
- utils: some useful functions to construct datasets, store them, create plots, train models etc. 

### Citation 
Please cite the original paper if you are using GraphSVX in your work. 
```
@inproceedings{duval2021graphsvx,
  title={GraphSVX: Shapley Value Explanations for Graph Neural Networks},
  author={Duval, Alexandre and Malliaros, Fragkiskos},
  booktitle={European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD)},
  year={2021}
}
```
