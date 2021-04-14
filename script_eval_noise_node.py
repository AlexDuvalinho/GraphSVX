""" script_eval_noise_node.py

    Evaluation of GraphSVX - filter noisy nodes
"""
import configs
import argparse
import numpy as np
import random
import time
from itertools import product
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.eval import filter_useless_nodes
from src.eval_multiclass import filter_useless_nodes_multiclass
from utils.io_utils import fix_seed


def main():

    args = configs.arg_parse()
    fix_seed(args.seed)
    node_indices = None

    start_time = time.time()

    if args.multiclass == False:
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

    else:
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


    end_time = time.time()
    print('Time: ', end_time - start_time)


if __name__ == "__main__":
    main()
