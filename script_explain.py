""" script_explain.py

    Derive explanations using GraphSVX 
"""

import argparse
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import configs
from utils.io_utils import fix_seed
from src.data import prepare_data
from src.explainers import GraphSVX
from src.train import evaluate, test


def main():

    args = configs.arg_parse()
    fix_seed(args.seed)

    # Load the dataset
    data = prepare_data(args.dataset, args.train_ratio,
                        args.input_dim, args.seed)

    # Load the model
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    model = torch.load(model_path)
    
    # Evaluate the model 
    if args.dataset in ['Cora', 'PubMed']:
        _, test_acc = evaluate(data, model, data.test_mask)
    else: 
        test_acc = test(data, model, data.test_mask)
    print('Test accuracy is {:.4f}'.format(test_acc))

    # Explain it with GraphSVX
    explainer = GraphSVX(data, model, args.gpu)

    # Distinguish graph classfication from node classification
    if args.dataset in ['Mutagenicity', 'syn6']:
        explanations = explainer.explain_graphs(args.indexes,
                                         args.hops,
                                         args.num_samples,
                                         args.info,
                                         args.multiclass,
                                         args.fullempty,
                                         args.S,
                                         'graph_classification',
                                         args.feat,
                                         args.coal,
                                         args.g,
                                         args.regu,
                                         True)
    else: 
        explanations = explainer.explain(args.indexes,
                                        args.hops,
                                        args.num_samples,
                                        args.info,
                                        args.multiclass,
                                        args.fullempty,
                                        args.S,
                                        args.hv,
                                        args.feat,
                                        args.coal,
                                        args.g,
                                        args.regu,
                                        True)

    print('Sum explanations: ', [np.sum(explanation) for explanation in explanations])
    print('Base value: ', explainer.base_values)

if __name__ == "__main__":
    main()
