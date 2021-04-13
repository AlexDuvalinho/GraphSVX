
GraphSVX: Shapley Value Explanations for Graph Neural Networks 

In this folder, we provide the code of GraphSVX, as well as the evaluation
pipeline defined in the paper, which we use to show the functioning of GraphSVX. 

If needed, install the required packages contained in requirements.txt

To run the code, type: python3 filename (below)
    - script_train.py: train models for all node classification and graph classification tasks. 
    - script_explain.py: explain node prediction or graph classification with GraphSVX, for any dataset
    - script_eval_noise.py: evaluate GraphSVX on noisy dataset and observe number of 
    noisy features/nodes included in explanations 
    - script_eval_gt.py: evaluate GraphSVX on synthetic datasets with a ground truth. 
All parameters are in the configs.py file, along with a small documentation. 

The structure of the code is as follows: 
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
