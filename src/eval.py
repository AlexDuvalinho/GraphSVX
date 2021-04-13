""" eval.py

    Evaluation 1 of the GraphSVX explainer - for pred class only
    Add noise features and neighbours to dataset
    Check how frequently they appear in explanations
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.nn import GNNExplainer as GNNE

from src.models import GAT, GCN
from src.train import accuracy, train_and_val
from src.utils import *
from src.data import (add_noise_features, add_noise_neighbours,
                      extract_test_nodes, prepare_data)
from src.explainers import (LIME, SHAP, GNNExplainer, GraphLIME, GraphSVX,
                            Greedy, Random)
from src.plots import plot_dist



###############################################################################
# Node classification on synthetic datasets
############################################################################

def eval_syn(data, model, args):
    """ Evaluate performance of explainer on synthetic
    datasets with a ground truth 

    Args:
        data (NameSpace): dataset information
        model: trained GNN model
        args (NameSpace): input arguments
    """
    # Define ground truth and test nodes for each dataset
    k = 4  # number of nodes for the shape introduced (house, cycle)
    K = 0
    if args.dataset == 'syn1':
        node_indices = list(range(400, 500, 5))
    elif args.dataset == 'syn2':
        node_indices = list(range(400, 425, 5)) + list(range(1100, 1125, 5))
    elif args.dataset == 'syn4':
        node_indices = list(range(511, 691, 6))  # (511, 571, 6)
        if args.hops == 3:
            k = 5
        else:
            K = 5
    elif args.dataset == 'syn5':
        node_indices = list(range(511, 654, 9))  # (511, 601, 9)
        #k = 7
        k = 5
        K = 7

    # GraphSHAP - assess accuracy of explanations
    graphsvx = GraphSVX(data, model, args.gpu)

    # Loop over test nodes
    accuracy = []
    diff_in_pred = []
    percentage_fidelity = []
    feat_accuracy = []
    for node_idx in node_indices:
        graphsvx_explanations = graphsvx.explain([node_idx],
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
                                                 )[0]

        # Keep only node explanations
        graphsvx_node_explanations = graphsvx_explanations[graphsvx.F:]

        # Derive ground truth from graph structure
        ground_truth = list(range(node_idx+1, node_idx+max(k, K)+1))

        # Retrieve top k elements indices form graphsvx_node_explanations
        l = list(graphsvx.neighbours).index(ground_truth[0])
        if args.info: 
            print('Importance:', np.sum(graphsvx_explanations[l:l+5]))
        #print('Importance:', np.sum(
        #    graphsvx_explanations[l:l+4]) / (np.sum(graphsvx_explanations)-0.01652819)) # base value

        if graphsvx.neighbours.shape[0] > k:
            i = 0
            val, indices = torch.topk(torch.tensor(
                graphsvx_node_explanations.T), k+1)
            # could weight importance based on val
            for node in graphsvx.neighbours[indices]:
                if node.item() in ground_truth:
                    i += 1
            # Sort of accruacy metric
            accuracy.append(i / k)

            if args.info: 
                print('There are {} from targeted shape among most imp. nodes'.format(i))

        # Look at importance distribution among features
        # Identify most important features and check if it corresponds to truly imp ones
        if args.dataset == 'syn2':
            graphsvx_feat_explanations = graphsvx_explanations[:graphsvx.F]
            print('Feature importance graphsvx',
                  graphsvx_feat_explanations.T)
            feat_accuracy.append(len(set(np.argsort(
                graphsvx_feat_explanations)[-2:]).intersection([0, 1])) / 2)

    print('Node Accuracy: {:.2f}, Feature Accuracy: {:.2f}'.format(np.mean(accuracy), \
                                                                    np.mean(feat_accuracy)))

###############################################################################
# Graph Classification Eval
############################################################################

def eval_Mutagenicity(data, model, args):
    """Evaluate GraphSVX on MUTAG dataset

    Args:
        data (NameSpace): pre-processed MUTAG dataset
        model (): GNN model
        args (argparse): all parameters
    """
    allgraphs = list(range(len(data.selected)))[100:120]
    accuracy = []
    for graph_idx in allgraphs:
        graphsvx = GraphSVX(data, model, args.gpu)
        graphsvx_explanations = graphsvx.explain_graphs([graph_idx],
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
                                                        regu=0,
                                                        vizu=False)[0]

        # Find ground truth in orginal data
        idexs = np.nonzero(data.edge_label_lists[graph_idx])[0].tolist()
        inter = []  # retrieve edge g.t. from above indexes of g.t.
        for i in idexs:
            inter.append(data.edge_lists[graph_idx][i])
        ground_truth = [item for sublist in inter for item in sublist]
        ground_truth = list(set(ground_truth))  # node g.t.

        # Find ground truth (nodes) for each graph
        k = len(ground_truth)  # Length gt

        # Retrieve top k elements indices form graphsvx_explanations
        if len(graphsvx.neighbours) >= k:
            i = 0
            val, indices = torch.topk(torch.tensor(
                graphsvx_explanations.T), k)
            # could weight importance based on val
            for node in torch.tensor(graphsvx.neighbours)[indices]:
                if node.item() in ground_truth:
                    i += 1
            # Sort of accruacy metric
            accuracy.append(i / k)
            print('acc:', i/k)
            print('indexes', indices)
            print('gt', ground_truth)

    print('Accuracy', accuracy)
    print('Mean accuracy', np.mean(accuracy))


def eval_syn6(data, model, args):
    """ Explain and evaluate syn6 dataset
    """
    # Define graphs used for evaluation
    allgraphs = np.nonzero(data.y).T[0].tolist()[:100]

    accuracy = []
    for graph_idx in allgraphs:
        graphsvx = GraphSVX(data, model, args.gpu)
        graphsvx_explanations = graphsvx.explain_graphs([graph_idx],
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
                                                        regu=0,
                                                        vizu=False)[0]

        # Retrieve ground truth (gt) from data
        preds = []
        reals = []
        ground_truth = list(range(20, 25))

        # Length gt
        k = len(ground_truth)

        # Retrieve top k elements indices form graphsvx_node_explanations
        if len(graphsvx.neighbours) >= k:
            i = 0
            val, indices = torch.topk(torch.tensor(
                graphsvx_explanations.T), k)
            # could weight importance based on val
            for node in torch.tensor(graphsvx.neighbours)[indices]:
                if node.item() in ground_truth:
                    i += 1
            # Sort of accruacy metric
            accuracy.append(i / k)
            print('acc:', i/k)

    print('accuracy', accuracy)
    print('mean', np.mean(accuracy))


###############################################################################
# Noisy features eval on real world datasets 
############################################################################

def filter_useless_features(args_dataset,
                            args_model,
                            args_explainers,
                            args_hops,
                            args_num_samples,
                            args_test_samples,
                            args_K,
                            args_prop_noise_feat,
                            node_indices,
                            info,
                            args_hv,
                            args_feat,
                            args_coal,
                            args_g,
                            args_multiclass,
                            args_regu,
                            args_gpu,
                            args_fullempty,
                            args_S, 
                            seed):
    """ Add noisy features to dataset and check how many are included in explanations
    The fewest, the better the explainer.

    Args:
        Arguments defined in argument parser of script_eval.py
    
    """

    # Define dataset 
    data = prepare_data(args_dataset, seed=seed)
    args_num_noise_feat = int(data.x.size(1) * args_prop_noise_feat)
    args_p = eval('EVAL1_' + data.name)['args_p']
    args_binary = eval('EVAL1_' + data.name)['args_binary']

    # Include noisy neighbours
    data, noise_feat = add_noise_features(
        data, num_noise=args_num_noise_feat, binary=args_binary, p=args_p)

    # Define training parameters depending on (model-dataset) couple
    hyperparam = ''.join(['hparams_', args_dataset, '_', args_model])
    param = ''.join(['params_', args_dataset, '_', args_model])

    # Define the model
    if args_model == 'GCN':
        model = GCN(input_dim=data.x.size(
            1), output_dim=data.num_classes, **eval(hyperparam))
    else:
        model = GAT(input_dim=data.x.size(
            1), output_dim=data.num_classes,  **eval(hyperparam))

    # Re-train the model on dataset with noisy features
    train_and_val(model, data, **eval(param))

    # Select random subset of nodes to eval the explainer on.
    if not node_indices:
        node_indices = extract_test_nodes(data, args_test_samples, seed)
    
    # Evaluate the model on test set
    model.eval()
    with torch.no_grad():
        log_logits = model(x=data.x, edge_index=data.edge_index)  
    test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
    print('Test accuracy is {:.4f}'.format(test_acc))

    # Derive predicted class for each test sample
    with torch.no_grad():
        true_confs, predicted_classes = log_logits.exp()[node_indices].max(dim=1)
    del log_logits

    # Adaptable K - top k explanations we look at for each node
    # Depends on number of existing features considered for GraphSVX
    if 'GraphSVX' in args_explainers:
        K = []
    else:
        K = [10]*len(node_indices)
    #for node_idx in node_indices:
    #	K.append(int(data.x[node_idx].nonzero().shape[0] * args_K))

    if args_regu == 0:
        args_regu = 1

    # Loop on the different explainers selected
    for c, explainer_name in enumerate(args_explainers):
        
        # Define explainer
        explainer = eval(explainer_name)(data, model, args_gpu)
        print('EXPLAINER: ', explainer_name)

        # count noisy features found in explanations 
        pred_class_num_noise_feats = []
        # count number of noisy features considered
        total_num_noise_feat_considered = []
        # count number of features   
        F = []

        # Loop on each test sample and store how many times do noise features appear among
        # K most influential features in our explanations
        j=0
        for node_idx in tqdm(node_indices, desc='explain node', leave=False):
            
            # Explanations via GraphSVX
            if explainer_name == 'GraphSVX':
                coefs = explainer.explain(
                                [node_idx],
                                args_hops,
                                args_num_samples,
                                info,
                                args_multiclass,
                                args_fullempty,
                                args_S,
                                args_hv,
                                args_feat,
                                args_coal,
                                args_g,
                                args_regu,
                                )
                # Look only at features coefficients 
                # Neighbours are irrelevant here
                coefs = coefs[0][:explainer.F]
                
                # Adaptable K
                if explainer.F > 100:
                    K.append(int(args_K * 100))
                else:
                    K.append( max(1, int(explainer.F * args_K)) )

                # Num_features_considered
                if args_feat == 'Null':
                    feat_idx = noise_feat[explainer.neighbours, :].mean(axis=0).nonzero()
                    num_noise_feat_considered = feat_idx.size()[0]

                # Consider all features (+ use expectation like below)
                elif args_feat == 'All':
                    num_noise_feat_considered = args_num_noise_feat

                # Consider only features whose aggregated value is different from expected one
                else:
                    # Stats dataset
                    var = noise_feat.std(axis=0)
                    mean = noise_feat.mean(axis=0)
                    # Feature intermediate rep
                    mean_subgraph = noise_feat[explainer.neighbours, :].mean(axis=0)
                    # Select relevant features only - (E-e,E+e)
                    mean_subgraph = torch.where(mean_subgraph > mean - 0.25*var, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
                    mean_subgraph = torch.where(mean_subgraph < mean + 0.25*var, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
                    feat_idx = (mean_subgraph == 100).nonzero()
                    num_noise_feat_considered = feat_idx.shape[0]
                    del mean, mean_subgraph, var
                
            else:
                coefs = explainer.explain(node_idx,
                                        args_hops,
                                        args_num_samples,
                                        info=False,
                                        multiclass=False
                                        )[:explainer.F]
                # All features are considered
                num_noise_feat_considered = args_num_noise_feat

            # Features considered 
            F.append(explainer.F)

            # Store indexes of K most important node features, for each class
            feat_indices = coefs.argsort()[-K[j]:].tolist()

            # Number of noisy features that appear in explanations - use index to spot them
            num_noise_feat = [idx for idx in feat_indices if idx > (explainer.F - num_noise_feat_considered)]

            # If node importance of top K features is unsignificant, discard 
            # Possible as we have importance informative measure, unlike others.
            if explainer_name == 'GraphSVX':
                explainable_part = true_confs[c] - \
                                explainer.base_values[c]
                num_noise_feat = [idx for idx in num_noise_feat if np.abs(coefs[idx]) > 0.05*np.abs(explainable_part)]
            
            # Count number of noisy that appear in explanations
            num_noise_feat = len(num_noise_feat)
            pred_class_num_noise_feats.append(num_noise_feat)

            # Return number of noisy features considered in this test sample
            total_num_noise_feat_considered.append(num_noise_feat_considered)

            j+=1

        print('Noisy features included in explanations: ',
                        sum(pred_class_num_noise_feats) )
        print('For the predicted class, there are {} noise features found in the explanations of {} test samples, an average of {} per sample'
                        .format(sum(pred_class_num_noise_feats), args_test_samples, sum(pred_class_num_noise_feats)/args_test_samples))

        print(pred_class_num_noise_feats)

        if sum(F) != 0:
            perc = 100 * sum(total_num_noise_feat_considered) / sum(F)
            print(
                'Proportion of considered noisy features among features: {:.2f}%'.format(perc))
        if sum(K) != 0:
            perc = 100 * sum(pred_class_num_noise_feats) / sum(K)
            print('Proportion of explanations showing noisy features: {:.2f}%'.format(perc))

        if sum(total_num_noise_feat_considered) != 0:
            perc = 100 * sum(pred_class_num_noise_feats) / (sum(total_num_noise_feat_considered))
            perc2 = 100 * (sum(K) - sum(pred_class_num_noise_feats)) / (sum(F) - sum(total_num_noise_feat_considered)) 
            print('Proportion of noisy features found in explanations vs normal features (among considered ones): {:.2f}% vs {:.2f}%, over considered features only'.format(
                perc, perc2))

        print('------------------------------------')

        # Plot of kernel density estimates of number of noisy features included in explanation
        # Do for all benchmarks (with diff colors) and plt.show() to get on the same graph
        plot_dist(pred_class_num_noise_feats, 
                    label=explainer_name, color=COLOURS[c])

    # Random explainer - plot estimated kernel density
    total_num_noise_feats = noise_feats_for_random(
        data, model, K, args_num_noise_feat, node_indices)
    save_path = 'results/eval1_feat'
    plot_dist(total_num_noise_feats, label='Random', color='y')

    # Store graph - with key params and time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    plt.savefig('results/eval1_feat_{}_{}_{}_{}_{}.pdf'.format(data.name,
                                                           args_coal, 
                                                           args_feat, 
                                                           args_hv, 
                                                           current_time))
    plt.close()
    # plt.show()


def noise_feats_for_random(data, model, K, args_num_noise_feat, node_indices):
    """ Random explainer

    Args: 
        K: number of most important features we look at 
        args_num_noise_feat: number of noisy features 
        node_indices: indices of test nodes 
    
    Returns:
        Number of times noisy features are provided as explanations 
    """
    # Loop on each test sample and store how many times do noise features appear among
    # K most influential features in our explanations
    pred_class_num_noise_feats = []

    for j, node_idx in enumerate(node_indices):

        # Use Random explainer
        explainer = Random(data.x.size(1), K[j])

        # Store indexes of K most important features, for each class
        feat_indices = explainer.explain()

        # Number of noisy features that appear in explanations - use index to spot them
        num_noise_feat = sum(
            idx < args_num_noise_feat for idx in feat_indices)

        pred_class_num_noise_feats.append(num_noise_feat)

    return pred_class_num_noise_feats


###############################################################################
# Noisy noise eval on real world datasets
############################################################################


def filter_useless_nodes(args_dataset,
                         args_model,
                         args_explainers,
                         args_hops,
                         args_num_samples,
                         args_test_samples,
                         args_K,
                         args_prop_noise_nodes,
                         args_connectedness,
                         node_indices,
                         info,
                         args_hv,
                         args_feat,
                         args_coal,
                         args_g,
                         args_multiclass,
                         args_regu,
                         args_gpu,
                         args_fullempty,
                         args_S,
                         seed):
    """ Add noisy neighbours to dataset and check how many are included in explanations
    The fewest, the better the explainer.

    Args:
        Arguments defined in argument parser of script_eval.py
    
    """

    # Define dataset
    data = prepare_data(args_dataset, seed=seed)

    # Select a random subset of nodes to eval the explainer on.
    if not node_indices:
        node_indices = extract_test_nodes(data, args_test_samples, seed)
    
    # Define number of noisy nodes according to dataset size
    args_num_noise_nodes = int(args_prop_noise_nodes * data.x.size(0))
    args_c = eval('EVAL1_' + data.name)['args_c']
    args_p = eval('EVAL1_' + data.name)['args_p']
    args_binary = eval('EVAL1_' + data.name)['args_binary']

    # Add noisy neighbours to the graph, with random features
    data = add_noise_neighbours(data, args_num_noise_nodes, node_indices,
                                binary=args_binary, p=args_p, 
                                connectedness=args_connectedness, c=args_c)

    # Define training parameters depending on (model-dataset) couple
    hyperparam = ''.join(['hparams_', args_dataset, '_', args_model])
    param = ''.join(['params_', args_dataset, '_', args_model])

    # Define the model
    if args_model == 'GCN':
        model = GCN(input_dim=data.x.size(
            1), output_dim=data.num_classes, **eval(hyperparam))
    else:
        model = GAT(input_dim=data.x.size(
            1), output_dim=data.num_classes, **eval(hyperparam))

    # Re-train the model on dataset with noisy features
    train_and_val(model, data, **eval(param))
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        log_logits = model(x=data.x, edge_index=data.edge_index)  # [2708, 7]
    test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
    print('Test accuracy is {:.4f}'.format(test_acc))

    # Derive predicted class for each test node
    with torch.no_grad():
        true_confs, predicted_classes = log_logits.exp()[node_indices].max(dim=1)
    del log_logits

    if args_regu == 1:
        args_regu = 0

    # Study attention weights of noisy nodes in GAT model - compare attention with explanations
    if str(type(model)) == "<class 'src.models.GAT'>":
        study_attention_weights(data, model, args_test_samples)

    # Do for several explainers
    for c, explainer_name in enumerate(args_explainers):

        print('EXPLAINER: ', explainer_name)	
        # Define the explainer
        explainer = eval(explainer_name)(data, model, args_gpu)

        # Loop on each test sample and store how many times do noisy nodes appear among
        # K most influential features in our explanations
        # count number of noisy nodes in explanations 
        pred_class_num_noise_neis = []
        # count number of noisy nodes in subgraph
        total_num_noisy_nei = []
        # Number of neigbours of v in subgraph
        total_neigbours = []  
        # Stores number of most important neighbours we look at, for each node 
        K = []
        # To retrieve the predicted class
        j = 0
        for node_idx in tqdm(node_indices, desc='explain node', leave=False):

            # Look only at coefficients for nodes (not node features)
            if explainer_name == 'Greedy':
                coefs = explainer.explain_nei(node_idx,
                                              args_hops,
                                              args_num_samples)

            elif explainer_name == 'GNNExplainer':
                _ = explainer.explain(node_idx,
                                      args_hops,
                                      args_num_samples)
                coefs = explainer.coefs

            else:
                # Explanations via GraphSVX
                coefs = explainer.explain([node_idx],
                                          args_hops,
                                          args_num_samples,
                                          info,
                                          args_multiclass,
                                          args_fullempty,
                                          args_S,
                                          args_hv,
                                          args_feat,
                                          args_coal,
                                          args_g,
                                          args_regu)
                coefs = coefs[0].T[explainer.F:]

            # Number of noisy nodes in the subgraph of node_idx
            num_noisy_nodes = len(
                [n_idx for n_idx in explainer.neighbours if n_idx >= data.x.size(0)-args_num_noise_nodes])

            # Number of neighbours in the subgraph
            total_neigbours.append(len(explainer.neighbours))

            # Adaptable K - vary according to number of nodes in the subgraph
            if len(explainer.neighbours) > 100:
                K.append(int(args_K * 100))
            else:
                K.append( max(1, int(args_K * len(explainer.neighbours))) )

            # Store indexes of K most important features, for each class
            nei_indices = coefs.argsort()[-K[j]:].tolist()

            # Number of noisy features that appear in explanations - use index to spot them
            noise_nei = [idx for idx in nei_indices if idx > (explainer.neighbours.shape[0] - num_noisy_nodes)]

            # If node importance of top K neighbours is unsignificant, discard 
            # Possible as we have importance informative measure, unlike others.
            if explainer_name == 'GraphSVX':
                explainable_part = true_confs[c] - \
                                explainer.base_values[c]
                noise_nei = [idx for idx in noise_nei if np.abs(coefs[idx]) > 0.05*np.abs(explainable_part)]
            
            num_noise_nei = len(noise_nei)
            pred_class_num_noise_neis.append(num_noise_nei)

            # Return number of noisy nodes adjacent to node of interest
            total_num_noisy_nei.append(num_noisy_nodes)

            j += 1

        print('Noisy neighbours included in explanations: ',
                        pred_class_num_noise_neis)

        print('There are {} noise neighbours found in the explanations of {} test samples, an average of {} per sample'
                        .format(sum(pred_class_num_noise_neis), args_test_samples, sum(pred_class_num_noise_neis)/args_test_samples))

        print('Proportion of explanations showing noisy neighbours: {:.2f}%'.format(
            100 * sum(pred_class_num_noise_neis) / sum(K)))

        perc = 100 * sum(pred_class_num_noise_neis) / (sum(total_num_noisy_nei))
        perc2 = 100 * (sum(K) - sum(pred_class_num_noise_neis)) \
        / (sum(total_neigbours) - sum(total_num_noisy_nei))
        print('Proportion of noisy neighbours found in explanations vs normal neighbours (in subgraph): {:.2f}% vs {:.2f}'.format(
            perc, perc2))

        print('Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(
            100 * sum(total_num_noisy_nei) / sum(total_neigbours)))

        print('Proportion of noisy neighbours found in explanations (entire graph): {:.2f}%'.format(
            100 * sum(pred_class_num_noise_neis) / (args_test_samples * args_num_noise_nodes)))
        
        print('------------------------------------')

        # Plot of kernel density estimates of number of noisy features included in explanation
        # Do for all benchmarks (with diff colors) and plt.show() to get on the same graph
        plot_dist(pred_class_num_noise_neis,
                    label=explainer_name, color=COLOURS[c])

    # Random explainer - plot estimated kernel density
    total_num_noise_neis = noise_nodes_for_random(
        data, model, K, node_indices, total_num_noisy_nei, total_neigbours)
    plot_dist(total_num_noise_neis, label='Random',
              color='y')
    
    # Store graph - with key params and time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    plt.savefig('results/eval1_node_{}_{}_{}_{}_{}.pdf'.format(data.name,
                                                           args_coal, 
                                                           args_feat, 
                                                           args_hv, 
                                                           current_time))
    plt.close()
    #plt.show()

    return total_num_noise_neis


def noise_nodes_for_random(data, model, K, node_indices, total_num_noisy_nei, total_neigbours):
    """ Random explainer (for neighbours)

    Args: 
        K: number of important features for each test sample
        total_neigbours: list of number of neighbours for node_indices
        total_num_noisy_nei: list of number of noisy features for node_indices
        node_indices: indices of test nodes 
    
    Returns:
        Number of times noisy features are provided as explanations 
    """
    pred_class_num_noise_neis = []

    for j, _ in enumerate(node_indices):

        # Use Random explainer - on neighbours (not features)
        explainer = Random(total_neigbours[j], K[j])

        # Store indexes of K most important features, for each class
        nei_indices = explainer.explain()

        # Number of noisy features that appear in explanations - use index to spot them
        num_noise_nei = sum(
                    idx >= (total_neigbours[j]-total_num_noisy_nei[j]) for idx in nei_indices)

        pred_class_num_noise_neis.append(num_noise_nei)

    return pred_class_num_noise_neis


def study_attention_weights(data, model, args_test_samples):
    """ Studies the attention weights of the GAT model
    """
    _, alpha, alpha_bis = model(data.x, data.edge_index, att=True)

    # remove self loops att
    edges, alpha1 = alpha[0][:, :-
                          (data.x.size(0))], alpha[1][:-(data.x.size(0)), :]
    alpha2 = alpha_bis[1][:-(data.x.size(0))]

    # Look at all importance coefficients of noisy nodes towards normal nodes
    att1 = []
    att2 = []
    for i in range(data.x.size(0) - args_test_samples, (data.x.size(0)-1)):
        ind = (edges == i).nonzero()
        for j in ind[:, 1]:
            att1.append(torch.mean(alpha1[j]))
            att2.append(alpha2[j][0])
    print('shape attention noisy', len(att2))

    # It looks like these noisy nodes are very important
    print('av attention',  (torch.mean(alpha1) + torch.mean(alpha2))/2)  # 0.18
    (torch.mean(torch.stack(att1)) + torch.mean(torch.stack(att2)))/2  # 0.32

    # In fact, noisy nodes are slightly below average in terms of attention received
    # Importance of interest: look only at imp. of noisy nei for test nodes
    print('attention 1 av. for noisy nodes: ',
            torch.mean(torch.stack(att1[0::2])))
    print('attention 2 av. for noisy nodes: ',
            torch.mean(torch.stack(att2[0::2])))

    return torch.mean(alpha[1], axis=1)

############################################################################

def eval_shap(args_dataset,
              args_model,
              args_test_samples,
              args_hops,
              args_K,
              args_num_samples,
              node_indices,
              info,
              args_hv,
              args_feat,
              args_coal,
              args_g,
              args_multiclass,
              args_regu,
              args_gpu,
              args_fullempty,
              args_S, 
              seed):
    """
    Compares SHAP and GraphSVX on graph based datasets
    Check if they agree on features'contribution towards prediction for several test samples
    """

    # Define dataset
    data = prepare_data(args_dataset, seed=10)

    # Select a random subset of nodes to eval the explainer on.
    if not node_indices:
        node_indices = extract_test_nodes(data, args_test_samples, seed)

    # Define training parameters depending on (model-dataset) couple
    hyperparam = ''.join(['hparams_', args_dataset, '_', args_model])
    param = ''.join(['params_', args_dataset, '_', args_model])

    # Define the model
    if args_model == 'GCN':
        model = GCN(input_dim=data.x.size(
            1), output_dim=data.num_classes, **eval(hyperparam))
    else:
        model = GAT(input_dim=data.x.size(
            1), output_dim=data.num_classes, **eval(hyperparam))

    # Re-train the model on dataset with noisy features
    train_and_val(model, data, **eval(param))

    # Store metrics
    iou = []
    prop_contrib_diff = []

    # Iterate over test samples
    for node_idx in tqdm(node_indices, desc='explain node', leave=False):

        # Define explainers we would like to compare
        graphshap = GraphSVX(data, model, args_gpu)
        shap = SHAP(data, model, args_gpu)

        # Explanations via GraphSVX
        graphshap_coefs = graphshap.explain([node_idx],
                                  args_hops,
                                  args_num_samples,
                                  info,
                                  args_multiclass,
                                  args_fullempty,
                                  args_S,
                                  args_hv,
                                  args_feat, # All
                                  args_coal, # Random or SmarerSoftRegu
                                  args_g, #  WLS
                                  args_regu) # 1
        graphshap_coefs = graphshap_coefs[0].T[:graphshap.F]

        shap_coefs = shap.explain(node_idx,
                                  args_hops,
                                  args_num_samples,
                                  info=False,
                                  multiclass=False
                                  )[:shap.F]

        # Consider node features only - for predicted class only
        true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
            node_idx].max(dim=0)

        # Need to apply regularisation

        # Proportional contribution
        prop_contrib_diff.append(np.abs( graphshap_coefs.sum(
        ) / np.abs(graphshap_coefs).sum() - shap_coefs.sum() / np.abs(shap_coefs).sum()))
        #print('GraphSVX proportional contribution to pred: {:.2f}'.format(graphshap_coefs.sum() / np.abs(graphshap_coefs).sum() ))
        #print('SHAP proportional contribution to pred: {:.2f}'.format(shap_coefs.sum() / np.abs(shap_coefs).sum() ))

        # Important features
        graphshap_feat_indices = np.abs(graphshap_coefs).argsort()[-10:].tolist()
        shap_feat_indices = np.abs(shap_coefs).argsort()[-10:].tolist()
        iou.append(len(set(graphshap_feat_indices).intersection(set(shap_feat_indices))
                       ) / len(set(graphshap_feat_indices).union(set(shap_feat_indices))))
        #print('Iou important features: ', iou)

    print('iou av:', np.mean(iou))
    print('difference in contibutions towards pred: ', np.mean(prop_contrib_diff))
