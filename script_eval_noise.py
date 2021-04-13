""" script_eval.py

    Evaluation and benchmark of GraphSVX explainer
"""

import argparse
import numpy as np
import random
import time
from itertools import product
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import configs
from utils.io_utils import fix_seed
from src.eval_multiclass import filter_useless_features_multiclass, filter_useless_nodes_multiclass
from src.eval import filter_useless_features, filter_useless_nodes, eval_shap



def main():

    args = configs.arg_parse()
    fix_seed(args.seed)
    node_indices = None
    
    start_time = time.time()


    if args.multiclass == False:
        
        # Noisy nodes
        filter_useless_nodes(args.dataset,
                             args.model,
                             args.node_explainers,
                             args.hops,
                             args.num_samples,
                             args.test_samples,
                             args.K,
                             args.prop_noise_nodes,
                             args.connectedness,
                             node_indices,
                             args.info,
                             args.hv,
                             args.feat,
                             args.coal,
                             args.g,
                             args.multiclass,
                             args.regu,
                             args.gpu,
                             args.fullempty,
                             args.S,
                             args.seed)
        
        # Noisy features
        filter_useless_features(args.dataset,
                                args.model,
                                args.feat_explainers,
                                args.hops,
                                args.num_samples,
                                args.test_samples,
                                args.K,
                                args.prop_noise_feat,
                                node_indices,
                                args.info,
                                args.hv,
                                args.feat,
                                args.coal,
                                args.g,
                                args.multiclass,
                                args.regu,
                                args.gpu,
                                args.fullempty,
                                args.S,
                                args.seed)
         
    else:
        # Neighbours
        filter_useless_nodes_multiclass(args.dataset,
                                        args.model,
                                        args.node_explainers,
                                        args.hops,
                                        args.num_samples,
                                        args.test_samples,
                                        args.prop_noise_nodes,
                                        args.connectedness,
                                        node_indices,
                                        5,
                                        args.info,
                                        args.hv,
                                        args.feat,
                                        args.coal,
                                        args.g,
                                        args.multiclass,
                                        args.regu,
                                        args.gpu,
                                        args.fullempty,
                                        args.S,
                                        args.seed)

        # Node features
        filter_useless_features_multiclass(args.dataset,
                                           args.model,
                                           args.feat_explainers,
                                           args.hops,
                                           args.num_samples,
                                           args.test_samples,
                                           args.prop_noise_feat,
                                           node_indices,
                                           5,
                                           args.info,
                                           args.hv,
                                           args.feat,
                                           args.coal,
                                           args.g,
                                           args.multiclass,
                                           args.regu,
                                           args.gpu,
                                           args.fullempty,
                                           args.S,
                                           args.seed)

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == "__main__":
    main()
