# The whole frame from GNN Explainer to get data and model
""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
import warnings
import torch
warnings.filterwarnings("ignore", category=FutureWarning)

import configs 
from src.data import prepare_data, selected_data
from src.eval import eval_Mutagenicity, eval_syn, eval_syn6
from utils.io_utils import fix_seed


def main(): 

    # Load a configuration
    args = configs.arg_parse()
    fix_seed(args.seed)

    # GPU or CPU
    if args.gpu:
        print("CUDA")
    else:
        print("Using CPU")

    # Load dataset
    data = prepare_data(args.dataset, args.train_ratio, args.input_dim, args.seed)
    
    # Load model 
    model_path = 'models/GCN_model_{}.pth'.format(args.dataset)
    model = torch.load(model_path)

    # Evaluate GraphSVX 
    if args.dataset == 'Mutagenicity':
        data = selected_data(data, args.dataset)
        eval_Mutagenicity(data, model, args)
    elif args.dataset == 'syn6': 
        eval_syn6(data, model, args)
    else: 
        eval_syn(data, model, args)
    
if __name__ == "__main__":
    main()
