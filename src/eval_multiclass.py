""" eval.py

    Evaluation 1 of the GraphSVX explainer
    Add noise features and neighbours to dataset
    Check how frequently they appear in explanations
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn import GNNExplainer as GNNE
from tqdm import tqdm

from src.data import (add_noise_features, add_noise_neighbours,
                      extract_test_nodes, prepare_data)
from src.explainers import (LIME, SHAP, GNNExplainer, GraphLIME, GraphSVX,
                            Greedy, Random)
from src.models import GAT, GCN
from src.plots import plot_dist
from src.train import accuracy, train_and_val
from src.utils import *

###############################################################################


def filter_useless_features_multiclass(args_dataset,
                                       args_model,
                                       args_explainers,
                                       args_hops,
                                       args_num_samples,
                                       args_test_samples,
                                       args_prop_noise_nodes,
                                       args_connectedness,
                                       node_indices,
                                       args_K,
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

    # Define dataset - include noisy features
    data = prepare_data(args_dataset, seed=10)
    args_num_noise_feat = int(data.x.size(1) * args_prop_noise_feat)
    args_p = eval('EVAL1_' + data.name)['args_p']
    args_binary = eval('EVAL1_' + data.name)['args_binary']
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

    # Evaluate the model - test set
    model.eval()
    with torch.no_grad():
        log_logits = model(x=data.x, edge_index=data.edge_index)  # [2708, 7]
    test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
    print('Test accuracy is {:.4f}'.format(test_acc))
    del log_logits

    # Loop on different explainers selected
    for c, explainer_name in enumerate(args_explainers):

        # Define explainer
        explainer = eval(explainer_name)(data, model, args_gpu)

        # count noisy features found in explanations for each test sample (for each class)
        total_num_noise_feats = []
        # count noisy features found in explanations for each test sample for class of interest
        pred_class_num_noise_feats = []
        # count number of noisy features considered for each test sample (for each class)
        total_num_noise_feat_considered = []
        F = []  # count number of non zero features for each test sample

        # Loop on each test sample and store how many times do noise features appear among
        # K most influential features in our explanations
        for node_idx in tqdm(node_indices, desc='explain node', leave=False):

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
                    args_regu)
                coefs = coefs[0].T[:explainer.F]

            # Explanations via GraphSVX
            else:
                coefs = explainer.explain(node_index=node_idx,
                                        hops=args_hops,
                                        num_samples=args_num_samples,
                                        info=False, 
                                        multiclass=True)
            

            # Check how many non zero features
            F.append(explainer.F)

            # Number of non zero noisy features
            # Dfferent for explainers with all features considered vs non zero features only (shap,graphshap)
            # if explainer.F != data.x.size(1)
            if explainer_name == 'GraphSVX' or explainer_name == 'SHAP':
                num_noise_feat_considered = len(
                    [val for val in noise_feat[node_idx] if val != 0])
            else:
                num_noise_feat_considered = args_num_noise_feat

            # Multilabel classification - consider all classes instead of focusing on the
            # class that is predicted by our model
            num_noise_feats = []
            true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
                node_idx].max(dim=0)

            for i in range(data.num_classes):

                # Store indexes of K most important node features, for each class
                feat_indices = np.abs(
                    coefs[:explainer.F, i]).argsort()[-args_K:].tolist()

                # Number of noisy features that appear in explanations - use index to spot them
                num_noise_feat = sum(
                    idx < num_noise_feat_considered for idx in feat_indices)
                num_noise_feats.append(num_noise_feat)

                # For predicted class only
                if i == predicted_class:
                    pred_class_num_noise_feats.append(num_noise_feat)

            # Return number of times noisy features are provided as explanations
            total_num_noise_feats.append(sum(num_noise_feats))

            # Return number of noisy features considered in this test sample
            total_num_noise_feat_considered.append(num_noise_feat_considered)

        if info:
            print('Noise features included in explanations: ',
                  total_num_noise_feats)
            print('There are {} noise features found in the explanations of {} test samples, an average of {} per sample'
                  .format(sum(total_num_noise_feats), args_test_samples, sum(total_num_noise_feats)/args_test_samples))

            # Number of noisy features found in explanation for the predicted class
            print(np.sum(pred_class_num_noise_feats) /
                  args_test_samples, 'for the predicted class only')

            perc = 100 * sum(total_num_noise_feat_considered) / np.sum(F)
            print(
                'Proportion of non-zero noisy features among non-zero features: {:.2f}%'.format(perc))

            perc = 100 * sum(total_num_noise_feats) / \
                (args_K * args_test_samples * data.num_classes)
            print(
                'Proportion of explanations showing noisy features: {:.2f}%'.format(perc))

            if sum(total_num_noise_feat_considered) != 0:
                perc = 100 * sum(total_num_noise_feats) / \
                    (sum(total_num_noise_feat_considered)*data.num_classes)
                perc2 = 100 * (args_K * args_test_samples * data.num_classes - sum(total_num_noise_feats)) / (
                    data.num_classes * (sum(F) - sum(total_num_noise_feat_considered)))
                print('Proportion of noisy features found in explanations vs normal features (among considered ones): {:.2f}% vs {:.2f}%, over considered features only'.format(
                    perc, perc2))

            print('------------------------------------')

        # Plot of kernel density estimates of number of noisy features included in explanation
        # Do for all benchmarks (with diff colors) and plt.show() to get on the same graph
        # plot_dist(total_num_noise_feats, label=explainer_name, color=COLOURS[c])
        plot_dist(total_num_noise_feats,
                    label=explainer_name, color=COLOURS[c])

    # Random explainer - plot estimated kernel density
    total_num_noise_feats = noise_feats_for_random(
        data, model, args_K, args_num_noise_feat, node_indices)
    save_path = 'results/eval1_feat'
    plot_dist(total_num_noise_feats, label='Random', color='y')

    plt.savefig('results/eval1_feat_{}'.format(data.name))
    # plt.show()


def noise_feats_for_random(data, model, args_K, args_num_noise_feat, node_indices):
    """ Random explainer

    Args: 
        args_K: number of important features
        args_num_noise_feat: number of noisy features
        node_indices: indices of test nodes 
    
    Returns:
        Number of times noisy features are provided as explanations 
    """

    # Use Random explainer
    explainer = Random(data.x.size(1), args_K)

    # Loop on each test sample and store how many times do noise features appear among
    # K most influential features in our explanations
    total_num_noise_feats = []
    pred_class_num_noise_feats = []

    for node_idx in node_indices:
        num_noise_feats = []
        true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
            node_idx].max(dim=0)

        for i in range(data.num_classes):
            # Store indexes of K most important features, for each class
            feat_indices = explainer.explain()

            # Number of noisy features that appear in explanations - use index to spot them
            num_noise_feat = sum(
                idx < args_num_noise_feat for idx in feat_indices)
            num_noise_feats.append(num_noise_feat)

            if i == predicted_class:
                pred_class_num_noise_feats.append(num_noise_feat)

        # Return this number => number of times noisy features are provided as explanations
        total_num_noise_feats.append(sum(num_noise_feats))

    #noise_feats = []
    # Do for each test sample
    # for node_idx in tqdm(range(args.test_samples), desc='explain node', leave=False):
    #	feat_indices = explainer.explain() # store indices of features provided as explanations
    #	noise_feat = (feat_indices >= INPUT_DIM[args.dataset]).sum() # check if they are noise features - not like this
    #	noise_feats.append(noise_feat)
    return total_num_noise_feats


###############################################################################


def filter_useless_nodes_multiclass(args_dataset,
                                    args_model,
                                    args_explainers,
                                    args_hops,
                                    args_num_samples,
                                    args_test_samples,
                                    args_prop_noise_nodes,
                                    args_connectedness,
                                    node_indices,
                                    args_K,
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
    data = prepare_data(args_dataset, seed=10)
    args_num_noise_nodes = int(args_prop_noise_nodes * data.x.size(0))
    args_c = eval('EVAL1_' + data.name)['args_c']
    args_p = eval('EVAL1_' + data.name)['args_p']
    args_binary = eval('EVAL1_' + data.name)['args_binary']

    # Select a random subset of nodes to eval the explainer on.
    if not node_indices:
        node_indices = extract_test_nodes(data, args_test_samples, seed)

    # Add noisy neighbours to the graph, with random features
    data = add_noise_neighbours(data, args_num_noise_nodes, node_indices,
                                binary=args_binary, p=args_p, connectedness=args_connectedness)

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

    model.eval()
    with torch.no_grad():
        log_logits = model(x=data.x, edge_index=data.edge_index)  # [2708, 7]
    test_acc = accuracy(log_logits[data.test_mask], data.y[data.test_mask])
    print('Test accuracy is {:.4f}'.format(test_acc))
    del log_logits

    # Study attention weights of noisy nodes in GAT model - compare attention with explanations
    if str(type(model)) == "<class 'src.models.GAT'>":
        study_attention_weights(data, model, args_test_samples)
    
    # Adaptable K - top k explanations we look at for each node
    # Depends on number of existing features/neighbours considered for GraphSVX
    # if 'GraphSVX' in args_explainers:
    # 	K = []
    # else:
    # 	K = [5]*len(node_indices)

    # Do for several explainers
    for c, explainer_name in enumerate(args_explainers):
        
        print('EXPLAINER: ', explainer_name)
        # Define the explainer
        explainer = eval(explainer_name)(data, model, args_gpu)

        # Loop on each test sample and store how many times do noisy nodes appear among
        # K most influential features in our explanations
        # 1 el per test sample - count number of noisy nodes in explanations
        total_num_noise_neis = []
        # 1 el per test sample - count number of noisy nodes in explanations for 1 class
        pred_class_num_noise_neis = []
        # 1 el per test sample - count number of noisy nodes in subgraph
        total_num_noisy_nei = []
        total_neigbours = []  # 1 el per test samples - number of neigbours of v in subgraph
        M = []  # 1 el per test sample - number of non zero features
        for node_idx in tqdm(node_indices, desc='explain node', leave=False):

            # Look only at coefficients for nodes (not node features)
            if explainer_name == 'Greedy':
                coefs = explainer.explain_nei(node_index=node_idx,
                                              hops=args_hops,
                                              num_samples=args_num_samples,
                                              info=False,
                                              multiclass=True)

            elif explainer_name == 'GNNExplainer':
                _ = explainer.explain(node_index=node_idx,
                                      hops=args_hops,
                                      num_samples=args_num_samples,
                                      info=False,
                                      multiclass=True)
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
            
            # if explainer.F > 50:
            # 	K.append(10)
            # else:
            # 	K.append(int(explainer.F * args_K))

            # Check how many non zero features
            M.append(explainer.M)

            # Number of noisy nodes in the subgraph of node_idx
            num_noisy_nodes = len(
                [n_idx for n_idx in explainer.neighbours if n_idx >= data.x.size(0)-args_num_noise_nodes])

            # Number of neighbours in the subgraph
            total_neigbours.append(len(explainer.neighbours))

            # Multilabel classification - consider all classes instead of focusing on the
            # class that is predicted by our model
            num_noise_neis = []  # one element for each class of a test sample
            true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
                node_idx].max(dim=0)

            for i in range(data.num_classes):

                # Store indexes of K most important features, for each class
                nei_indices = np.abs(coefs[:, i]).argsort()[-args_K:].tolist()

                # Number of noisy features that appear in explanations - use index to spot them
                num_noise_nei = sum(
                    idx >= (explainer.neighbours.shape[0] - num_noisy_nodes) for idx in nei_indices)
                num_noise_neis.append(num_noise_nei)

                if i == predicted_class:
                    #nei_indices = coefs[:,i].argsort()[-args_K:].tolist()
                    #num_noise_nei = sum(idx >= (explainer.neighbours.shape[0] - num_noisy_nodes) for idx in nei_indices)
                    pred_class_num_noise_neis.append(num_noise_nei)

            # Return this number => number of times noisy neighbours are provided as explanations
            total_num_noise_neis.append(sum(num_noise_neis))
            # Return number of noisy nodes adjacent to node of interest
            total_num_noisy_nei.append(num_noisy_nodes)

        if info:
            print('Noisy neighbours included in explanations: ',
                  total_num_noise_neis)

            print('There are {} noise neighbours found in the explanations of {} test samples, an average of {} per sample'
                  .format(sum(total_num_noise_neis), args_test_samples, sum(total_num_noise_neis)/args_test_samples))

            print(np.sum(pred_class_num_noise_neis) /
                  args_test_samples, 'for the predicted class only')

            print('Proportion of explanations showing noisy neighbours: {:.2f}%'.format(
                100 * sum(total_num_noise_neis) / (args_K * args_test_samples * data.num_classes)))

            perc = 100 * sum(total_num_noise_neis) / (args_test_samples *
                                                      args_num_noise_nodes * data.num_classes)
            perc2 = 100 * ((args_K * args_test_samples * data.num_classes) -
                           sum(total_num_noise_neis)) / (np.sum(M) - sum(total_num_noisy_nei))
            print('Proportion of noisy neighbours found in explanations vs normal features: {:.2f}% vs {:.2f}'.format(
                perc, perc2))

            print('Proportion of nodes in subgraph that are noisy: {:.2f}%'.format(
                100 * sum(total_num_noisy_nei) / sum(total_neigbours)))

            print('Proportion of noisy neighbours in subgraph found in explanations: {:.2f}%'.format(
                100 * sum(total_num_noise_neis) / (sum(total_num_noisy_nei) * data.num_classes)))

        # Plot of kernel density estimates of number of noisy features included in explanation
        # Do for all benchmarks (with diff colors) and plt.show() to get on the same graph
        total_num_noise_neis = [item/data.num_classes for item in total_num_noise_neis]
        plot_dist(total_num_noise_neis,
                    label=explainer_name, color=COLOURS[c])
        # else:  # consider only predicted class
        # 	plot_dist(pred_class_num_noise_neis,
        # 			  label=explainer_name, color=COLOURS[c])

    # Random explainer - plot estimated kernel density
    total_num_noise_neis = noise_nodes_for_random(
        data, model, args_K, args_num_noise_nodes, node_indices)
    
    total_num_noise_neis= [item/data.num_classes for item in total_num_noise_neis]
    plot_dist(total_num_noise_neis, label='Random',
              color='y')

    plt.savefig('results/eval1_node_{}'.format(data.name))
    #plt.show()

    return total_num_noise_neis


def noise_nodes_for_random(data, model, args_K, args_num_noise_nodes, node_indices):
    """ Random explainer (for neighbours)

    Args: 
        args_K: number of important features
        args_num_noise_feat: number of noisy features
        node_indices: indices of test nodes 
    
    Returns:
        Number of times noisy features are provided as explanations 
    """

    # Use Random explainer - on neighbours (not features)
    explainer = Random(data.x.size(0), args_K)

    # Store number of noisy neighbours found in explanations (for all classes and predicted class)
    total_num_noise_neis = []
    pred_class_num_noise_neis = []

    # Check how many noisy neighbours are included in top-K explanations
    for node_idx in node_indices:
        num_noise_neis = []
        true_conf, predicted_class = model(x=data.x, edge_index=data.edge_index).exp()[
            node_idx].max(dim=0)

        for i in range(data.num_classes):
            # Store indexes of K most important features, for each class
            nei_indices = explainer.explain()

            # Number of noisy features that appear in explanations - use index to spot them
            num_noise_nei = sum(
                idx > (data.x.size(0)-args_num_noise_nodes) for idx in nei_indices)
            num_noise_neis.append(num_noise_nei)

            if i == predicted_class:
                pred_class_num_noise_neis.append(num_noise_nei)

        # Return this number => number of times noisy features are provided as explanations
        total_num_noise_neis.append(sum(num_noise_neis))

    #noise_feats = []
    # Do for each test sample
    # for node_idx in tqdm(range(args.test_samples), desc='explain node', leave=False):
    #	feat_indices = explainer.explain() # store indices of features provided as explanations
    #	noise_feat = (feat_indices >= INPUT_DIM[args.dataset]).sum() # check if they are noise features - not like this
    #	noise_feats.append(noise_feat)
    return total_num_noise_neis


def study_attention_weights(data, model, args_test_samples):
    """
        Studies the attention weights of the GAT model
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
