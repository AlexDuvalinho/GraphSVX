""" explainers.py

    Define the different explainers: GraphSVX and baselines
"""
# Import packages
import random
import time
from copy import deepcopy
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.special
import torch
import torch_geometric
from tqdm import tqdm
from sklearn.linear_model import (LassoLars, Lasso,
                                  LinearRegression, Ridge)
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GNNExplainer as GNNE
from torch_geometric.nn import MessagePassing

from src.models import LinearRegressionModel
from src.plots import (denoise_graph, k_hop_subgraph, log_graph,
                       visualize_subgraph, custom_to_networkx)



class GraphSVX():

    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        self.neighbours = None  #  nodes considered
        self.F = None  # number of features considered
        self.M = None  # number of features and nodes considered
        self.base_values = []

        self.model.eval()

    ################################
    # Core function - explain 
    ################################

    def explain(self,
                node_indexes=[0],
                hops=2,
                num_samples=10,
                info=True,
                multiclass=False,
                fullempty=None,
                S=3,
                args_hv='compute_pred',
                args_feat='Expectation',
                args_coal='Smarter',
                args_g='WLS',
                regu=None,
                vizu=False):
        """ Explain prediction for a given node - GraphSVX method

        Args:
            node_indexes (list, optional): indexes of the nodes of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph 
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSVX's new dataset. 
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings. 
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase 
            args_hv (str, optional): strategy used to convert simplified input z to original
                                                    input space z'
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            vizu (bool, optional): creates vizualisation or not

        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """
        # Time
        start = time.time()

        # Explain several nodes sequentially 
        phi_list = []
        for node_index in node_indexes:

            # Compute true prediction for original instance via explained GNN model
            if self.gpu: 
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                with torch.no_grad():
                    true_conf, true_pred = self.model(
                        self.data.x.cuda(), 
                        self.data.edge_index.cuda()).exp()[node_index].max(dim=0)
            else: 
                with torch.no_grad():
                    true_conf, true_pred = self.model(
                        self.data.x, 
                        self.data.edge_index).exp()[node_index].max(dim=0)

            # --- Node selection ---
            # Investigate k-hop subgraph of the node of interest (v)
            self.neighbours, _, _, edge_mask =\
                torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                    num_hops=hops,
                                                    edge_index=self.data.edge_index)
            
            # Retrieve 1-hop neighbours of v
            one_hop_neighbours, _, _, _ =\
                torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                    num_hops=1,
                                                    edge_index=self.data.edge_index)
            # Stores the indexes of the neighbours of v (+ index of v itself)

            # Remove node v index from neighbours and store their number in D
            self.neighbours = self.neighbours[self.neighbours != node_index]
            D = self.neighbours.shape[0]

            # --- Feature selection ---
            if args_hv == 'compute_pred_subgraph':
                feat_idx, discarded_feat_idx = self.feature_selection_subgraph(node_index, args_feat)
                # Also solve incompatibility due to overlap feat/node importance
                if args_hv == 'SmarterSeparate' or args_hv == 'NewSmarterSeparate':
                    print('Incompatibility: user Smarter sampling instead')
                    args_hv = 'Smarter' 
            else: 
                feat_idx, discarded_feat_idx = self.feature_selection(node_index, args_feat)

            # M: total number of features + neighbours considered for node v
            if regu==1 or D==0: 
                D=0
                print('Explainations only consider node features')
            if regu==0 or self.F==0:
                self.F=0
                print('Explainations only consider graph structure')
            self.M = self.F+D

            # Def range of endcases considered
            args_K = S

            # --- MASK GENERATOR --- 
            # Generate binary samples z' representing coalitions of nodes and features
            z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, info, regu)

            # Discard full and empty coalition if specified
            if fullempty:
                weights[weights == 1000] = 0

            # --- GRAPH GENERATOR ---
            # Create dataset (z, f(GEN(z'))), stored as (z_, fz)
            # Retrieve z' from z and x_v, then compute f(z')
            fz = eval('self.' + args_hv)(node_index, num_samples, D, z_,
                                        feat_idx, one_hop_neighbours, args_K, args_feat,
                                        discarded_feat_idx, multiclass, true_pred)

            # --- EXPLANATION GENERATOR --- 
            # Train Surrogate Weighted Linear Regression - learns shapley values
            phi, base_value = eval('self.' + args_g)(z_,
                                                    weights, fz, multiclass, info)

            # Rescale
            if type(regu) == int and not multiclass:
                expl = (true_conf.cpu() - base_value).detach().numpy()
                phi[:self.F] = (regu * expl / sum(phi[:self.F])) * phi[:self.F]
                phi[self.F:] = ((1-regu) * expl /
                                sum(phi[self.F:])) * phi[self.F:]

            # Print information
            if info:
                self.print_info(D, node_index, phi, feat_idx,
                                true_pred, true_conf, base_value, multiclass)

            # Visualise
            if vizu:
                self.vizu(edge_mask, node_index, phi,
                        true_pred, hops, multiclass)

            # Time
            end = time.time()
            if info:
                print('Time: ', end - start)

            # Append explanations for this node to list of expl.
            phi_list.append(phi)
            self.base_values.append(base_value)

        return phi_list

    def explain_graphs(self,
                       graph_indices=[0],
                       hops=2,
                       num_samples=10,
                       info=True,
                       multiclass=False,
                       fullempty=None,
                       S=3,
                       args_hv='compute_pred',
                       args_feat='Expectation',
                       args_coal='Smarter',
                       args_g='WLS',
                       regu=None,
                       vizu=False):
        """ Explains prediction for a graph classification task - GraphSVX method

        Args:
            node_indexes (list, optional): indexes of the nodes of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph 
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSVX's new dataset. 
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings. 
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase 
            args_hv (str, optional): strategy used to convert simplified input z to original
                                                    input space z'
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            vizu (bool, optional): creates vizualisation or not

        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """

        # Time
        start = time.time()

        # --- Explain several nodes iteratively ---
        phi_list = []
        for graph_index in graph_indices:

            # Compute true prediction for original instance via explained GNN model
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                with torch.no_grad():
                    true_conf, true_pred = self.model(self.data.x.cuda(),
                        self.data.edge_index.cuda()).exp()[graph_index,:].max(dim=0)
            else:
                with torch.no_grad():
                    true_conf, true_pred = self.model(self.data.x,
                        self.data.edge_index).exp()[graph_index,:].max(dim=0)

            # Remove node v index from neighbours and store their number in D
            self.neighbours = list(
                range(int(self.data.edge_index.shape[1] - np.sum(np.diag(self.data.edge_index[graph_index])))))
            D = len(self.neighbours)

            # Total number of features + neighbours considered for node v
            self.F = 0
            self.M = self.F+D

            # Def range of endcases considered
            args_K = S

            # --- MASK GENERATOR ---
            z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, info, regu)
            
            # Discard full and empty coalition if specified
            if fullempty:
                weights[(weights == 1000).nonzero()] = 0

            # --- GRAPH GENERATOR ---
            # Create dataset (z, f(GEN(z'))), stored as (z_, fz)
            # Retrieve z' from z and x_v, then compute f(z')
            fz = self.graph_classification(
                graph_index, num_samples, D, z_, args_K, args_feat, true_pred)

            # --- EXPLANATION GENERATOR --- 
            # Train Surrogate Weighted Linear Regression - learns shapley values
            phi, base_value = eval('self.' + args_g)(z_, weights, fz,
                                                     multiclass, info)

            phi_list.append(phi)
            self.base_values.append(base_value)

            return phi_list


    ################################
    # Feature selector
    ################################

    def feature_selection(self, node_index, args_feat):
        """ Select features who truly impact prediction
        Others will receive a 0 shapley value anyway 

        Args:
            node_index (int): node index
            args_feat (str): strategy utilised to select 
                                important featutres

        Returns:
            [tensor]: list of important features' index
            [tensor]: list of discarded features' index
        """
        
        #Only consider relevant features in explanations
        discarded_feat_idx = []
        if args_feat == 'All':
            # Select all features 
            self.F = self.data.x[node_index, :].shape[0]
            feat_idx = torch.unsqueeze(
                torch.arange(self.data.num_nodes), 1)
        elif args_feat == 'Null':
            # Select features whose value is non-null
            feat_idx = self.data.x[node_index, :].nonzero()
            self.F = feat_idx.size()[0]
        else:
            # Select features whose value is different from dataset mean value
            std = self.data.x.std(axis=0)
            mean = self.data.x.mean(axis=0)
            mean_subgraph = self.data.x[node_index, :]
            mean_subgraph = torch.where(mean_subgraph >= mean - 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            mean_subgraph = torch.where(mean_subgraph <= mean + 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            feat_idx = (mean_subgraph == 100).nonzero()
            discarded_feat_idx = (mean_subgraph != 100).nonzero()
            self.F = feat_idx.shape[0]

        return feat_idx, discarded_feat_idx

    def feature_selection_subgraph(self, node_index, args_feat):
        """ Similar to feature_selection (above)
        but considers the feature vector in the subgraph around v 
        instead of the feature of v
        """
        # Specific case: features in subgraph
        # Determine features and neighbours whose importance is investigated
        discarded_feat_idx = []
        if args_feat == 'All':
            # Consider all features - no selection 
            self.F = self.data.x[node_index, :].shape[0]
            feat_idx = torch.unsqueeze(
                torch.arange(self.data.num_nodes), 1)
        elif args_feat == 'Null':
            # Consider only non-zero entries in the subgraph of v
            feat_idx = self.data.x[self.neighbours, :].mean(
                axis=0).nonzero()
            self.F = feat_idx.size()[0]
        else:
            # Consider all features away from its mean value
            std = self.data.x.std(axis=0)
            mean = self.data.x.mean(axis=0)
            # Feature intermediate rep
            mean_subgraph = torch.mean(self.data.x[self.neighbours, :], dim=0)
            # Select relevant features only - (E-e,E+e)
            mean_subgraph = torch.where(mean_subgraph >= mean - 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            mean_subgraph = torch.where(mean_subgraph <= mean + 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            feat_idx = (mean_subgraph == 100).nonzero()
            discarded_feat_idx = (mean_subgraph != 100).nonzero()
            self.F = feat_idx.shape[0]
            del mean, mean_subgraph, std

        return feat_idx, discarded_feat_idx


    ################################
    # Mask generator
    ################################

    def mask_generation(self, num_samples, args_coal, args_K, D, info, regu):
            """ Applies selected mask generator strategy 

            Args:
                num_samples (int): number of samples for GraphSVX 
                args_coal (str): mask generator strategy 
                args_K (int): size param for indirect effect 
                D (int): number of nodes considered after selection
                info (bool): print information or not 
                regu (int): balances importance granted to nodes and features

            Returns:
                [tensor] (num_samples, M): dataset of samples/coalitions z' 
                [tensor] (num_samples): vector of kernel weights corresponding to samples 
            """
            if args_coal == 'SmarterSeparate' or args_coal == 'NewSmarterSeparate':
                weights = torch.zeros(num_samples, dtype=torch.float64)
                if self.F==0 or D==0:
                    num = int(num_samples * self.F/self.M)
                elif regu != None:
                    num = int(num_samples * regu)
                    #num = int( num_samples * ( self.F/self.M + ((regu - 0.5)/0.5)  * (self.F/self.M) ) )    
                else: 
                    num = int(0.5* num_samples/2 + 0.5 * num_samples * self.F/self.M)
                # Features only
                z_bis = eval('self.' + args_coal)(num, args_K, 1)  
                z_bis = z_bis[torch.randperm(z_bis.size()[0])]
                s = (z_bis != 0).sum(dim=1)
                weights[:num] = self.shapley_kernel(s, self.F)
                z_ = torch.zeros(num_samples, self.M)
                z_[:num, :self.F] = z_bis
                # Node only
                z_bis = eval('self.' + args_coal)(
                    num_samples-num, args_K, 0)  
                z_bis = z_bis[torch.randperm(z_bis.size()[0])]
                s = (z_bis != 0).sum(dim=1)
                weights[num:] = self.shapley_kernel(s, D)
                z_[num:, :] = torch.ones(num_samples-num, self.M)
                z_[num:, self.F:] = z_bis

            else:
                # If we choose to sample all possible coalitions
                if args_coal == 'All':
                    num_samples = min(10000, 2**self.M)

                # Coalitions: sample num_samples binary vectors of dimension M
                z_ = eval('self.' + args_coal)(num_samples, args_K, regu)

                # Shuffle them 
                z_ = z_[torch.randperm(z_.size()[0])]

                # Compute |z| for each sample z: number of non-zero entries
                s = (z_ != 0).sum(dim=1)

                # GraphSVX Kernel: define weights associated with each sample 
                weights = self.shapley_kernel(s, self.M)
                
            return z_, weights

    def NewSmarterSeparate(self, num_samples, args_K, regu):
        """Default mask sampler
        Generates feature mask and node mask independently
        Favours masks with a high weight + smart space allocation algorithm

        Args:
            num_samples (int): number of masks desired 
            args_K (int): maximum size of masks favoured
            regu (binary): nodes or features 

        Returns:
            tensor: dataset of samples
        """
        if regu == None:
            z_ = self.Smarter(num_samples, args_K, regu)
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            M = self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M) # case k = 0 
            i = 2
            k = 1
            P = num_samples * 9/10
            while i < P and k <= min(args_K, M-1): 

                # All coalitions of order k can be sampled 
                if i + 2 * scipy.special.comb(M,k) <= P:
                    L = list(combinations(range(M), k))
                    for c in L:
                        z_[i,c] = torch.zeros(k)
                        z_[i+1,c] = torch.ones(k)
                        i += 2
            
                # All coalitions of order k cannot be sampled
                else: 
                    weight = torch.ones(M)
                    L = list(combinations(range(M), k))
                    random.shuffle(L)
                    while i < min(P, len(L)):
                        cw = torch.tensor([sum(weight[list(c)]) for c in L])
                        c_idx = torch.argmax(cw).item()
                        c = list(L[c_idx])
                        p = float(random.randint(0,1))
                        z_[i,:] = torch.tensor(p).repeat(M)
                        z_[i,c] =  torch.tensor(1-p).repeat(len(c))
                        weight[list(c)] = torch.tensor([1/(1+1/el.item()) for el in weight[list(c)]])
                        i += 1
                    k += 1
            
            # Random coal
            z_[i:, :] = torch.empty( max(0, num_samples-i), M).random_(2)
            return z_

       # Favour features - special coalitions don't study node's effect
        elif regu < 0.5:
            M = self.M - self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M) # case k = 0 
            i = 2
            k = 1
            P = int(num_samples * 9/10)
            while i < P and k <= min(args_K, M-1): 

                # All coalitions of order k can be sampled 
                if i + 2 * scipy.special.comb(M,k) <= P:
                    L = list(combinations(range(M), k))
                    for c in L:
                        z_[i,c] = torch.zeros(k)
                        z_[i+1,c] = torch.ones(k)
                        i += 2
                    k += 1
            
                # All coalitions of order k cannot be sampled
                else: 
                    weight = torch.ones(M)
                    L = list(combinations(range(M), k))
                    random.shuffle(L)
                    while i < min(P, len(L)):
                        cw = torch.tensor([sum(weight[list(c)]) for c in L])
                        c_idx = torch.argmax(cw).item()
                        c = list(L[c_idx])
                        p = float(random.randint(0,1))
                        z_[i,:] = torch.tensor(p).repeat(M)
                        z_[i,c] =  torch.tensor(1-p).repeat(len(c))
                        weight[list(c)] = torch.tensor([1/(1+1/el.item()) for el in weight[list(c)]])
                        i += 1
                    k += 1
            
            # Random coal
            z_[i:, :] = torch.empty( max(0, num_samples-i), M).random_(2)
            return z_

    def SmarterSeparate(self, num_samples, args_K, regu):
        """Default mask sampler
        Generates feature mask and node mask independently
        Favours masks with a high weight + efficient space allocation algorithm

        Args:
            num_samples (int): number of masks desired 
            args_K (int): maximum size of masks favoured
            regu (binary): nodes or features 

        Returns:
            tensor: dataset of samples
        """
        if regu == None:
            z_ = self.Smarter(num_samples, args_K, regu)
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            # Define empty and full coalitions
            M = self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M)
            # z_[1, :] = torch.empty(1, self.M).random_(2)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * M < num_samples and k == 1:
                    z_[i:i+M, :] = torch.ones(M, M)
                    z_[i:i+M, :].fill_diagonal_(0)
                    z_[i+M:i+2*M, :] = torch.zeros(M, M)
                    z_[i+M:i+2*M, :].fill_diagonal_(1)
                    i += 2 * M
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = i + 9*(num_samples - i)//10
                    #samp = num_samples
                    while i < samp and k <= min(args_K, M):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(M), k))
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                    return z_
            return z_

        # Favour neighbour
        else:
            # Define empty and full coalitions
            M = self.M - self.F
            # self.F = 0
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * M < num_samples and k == 1:
                    z_[i:i+M, :] = torch.ones(M, M)
                    z_[i:i+M, :].fill_diagonal_(0)
                    z_[i+M:i+2*M, :] = torch.zeros(M, M)
                    z_[i+M:i+2*M, :].fill_diagonal_(1)
                    i += 2 * M
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    #samp = i + 9*(num_samples - i)//10
                    samp = num_samples
                    while i < samp and k <= min(args_K, M):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(0, M), k))
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                    return z_
            return z_

    def Smarter(self, num_samples, args_K, *unused):
        """ Smart Mask generator
        Nodes and features are considered together but separately

        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling 

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        i = 2
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 9*(num_samples - i)//10
                while i < samp and k <= args_K:
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(self.F), k)) + \
                        list(combinations(range(self.F, self.M), k))
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples-i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples-i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_
        return z_

    def Smart(self, num_samples, args_K, *unused):
        """ Sample coalitions cleverly 
        Favour coalition with height weight - no distinction nodes/feat

        Args:
            num_samples (int): total number of coalitions z_
            args_K (int): max size of coalitions favoured

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        k = 1
        i = 2
        while i < num_samples:
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1
            elif k == 1:
                M = list(range(self.M))
                random.shuffle(M)
                for j in range(self.M):
                    z_[i, M[j]] = torch.zeros(1)
                    i += 1
                    if i == num_samples:
                        return z_
                    z_[i, M[j]] = torch.ones(1)
                    i += 1
                    if i == num_samples:
                        return z_
                k += 1
            elif k < args_K:
                samp = i + 4*(num_samples - i)//5
                M = list(combinations(range(self.M), k))[:samp-i+1]
                random.shuffle(M)
                for j in range(len(M)):
                    z_[i, M[j][0]] = torch.tensor(0)
                    z_[i, M[j][1]] = torch.tensor(0)
                    i += 1
                    if i == samp:
                        z_[i:, :] = torch.empty(
                            num_samples-i, self.M).random_(2)
                        return z_
                    z_[i, M[j][0]] = torch.tensor(1)
                    z_[i, M[j][1]] = torch.tensor(1)
                    i += 1
                    if i == samp:
                        z_[i:, :] = torch.empty(
                            num_samples-i, self.M).random_(2)
                        return z_
                k += 1
            else:
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_

        return z_

    def Random(self, num_samples, *unused):
        """Sample masks randomly 

        """
        z_ = torch.empty(num_samples, self.M).random_(2)
        return z_

    def All(self, num_samples, *unsused):
        """Sample all possible 2^{F+N} coalitions (unordered, without replacement)

        Args:
            num_samples (int): 2^{M+N} or boundary we fixed (20,000)

        [tensor]: dataset (2^{M+N} x self.M) where each row is in {0,1}^F x {0,1}^D
        """
        z_ = torch.zeros(num_samples, self.M)
        i = 0
        try:
            for k in range(0, self.M+1):
                L = list(combinations(range(0, self.M), k))
                for j in range(len(L)):
                    z_[i, L[j]] = torch.ones(k)
                    i += 1
        except IndexError:  # deal with boundary
            return z_
        return z_

    ################################
    # GraphSVX kernel
    ################################

    def shapley_kernel(self, s, M):
        """ Computes a weight for each newly created sample 

        Args:
            s (tensor): contains dimension of z for all instances
                (number of features + neighbours included)
            M (tensor): total number of features/nodes in dataset

        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shapley_kernel = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                # Enforce high weight on full/empty coalitions
                shapley_kernel.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                # Treat specific case - impossible computation
                shapley_kernel.append(1/ (M**2))
            else:
                shapley_kernel.append(
                    (M-1)/(scipy.special.binom(M, a)*a*(M-a)))

        shapley_kernel = np.array(shapley_kernel)
        shapley_kernel = np.where(shapley_kernel<1.0e-40, 1.0e-40,shapley_kernel)
        return torch.tensor(shapley_kernel)

    ################################
    # Graph generator + compute f(z')
    ################################

    def compute_pred_subgraph(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Features in subgraph

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z') - probability of belonging to each target classes, for all samples z'
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        # Create networkx graph
        G = custom_to_networkx(self.data)
        G = G.subgraph(self.neighbours.tolist() + [node_index])

        # Define an "average" feature vector - for discarded features
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.data.x.mean(dim=0)
            # Change here for contrastive explanations
            # av_feat_values = self.data.x[402]
            # or random feature vector made of random value across each col of X

        # Init dict for nodes and features not sampled
        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z
        for i in range(num_samples):

            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)
        # classes_labels = torch.zeros(num_samples)
        # pred_confidence = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z from z
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            positions = []
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            # Special case (0 node, k feat) 
            # Consider only feat. influence if too few nei included
            if D - len(ex_nei) >= min(self.F - len(ex_feat), args_K):
                A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest - excluded and discarded features
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]
                
                # End approximation
                # if args_feat == 'Expectation':
                #    for val in discarded_feat_idx:
                #        X[self.neighbours, val] = av_feat_values[val].repeat(D)

            # Special case - consider only nei. influence if too few feat included
            if self.F - len(ex_feat) < min(D - len(ex_nei), args_K):
                # Don't set features = Exp or 0 in the whole subgraph, only for v.
            
                # Indirect effect
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                for incl_nei in included_nei:
                    paths = list(nx.all_shortest_paths(G, source=node_index, target=incl_nei))
                    np.random.shuffle(paths)
                    len_paths = [len(set(path[1:-1]).intersection(ex_nei)) for path in paths]
                    if min(len_paths) == 0: 
                        pass
                    else: 
                        path = paths[np.argmin(len_paths)]
                        for n in range(1, len(path)-1):
                            A = torch.cat((A, torch.tensor(
                                            [[path[n-1]], [path[n]]])), dim=-1)
                            X[path[n], :] = av_feat_values

            # Usual case - exclude features for the whole subgraph
            else:
                for val in ex_feat:
                    X[self.neighbours, val] = av_feat_values[val].repeat(len(self.neighbours))

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(X.cuda(), A.cuda()).exp()[node_index]
            else: 
                with torch.no_grad():
                    proba = self.model(X, A).exp()[node_index]

            # Store final class prediction and confience level
            # pred_confidence[key], classes_labels[key] = torch.topk(proba, k=1)

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

        return fz

    def compute_pred(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Standard method

        Args: 
                Variables are defined exactly as defined in explainer function 

        Returns: 
                (tensor): f(z') - probability of belonging to each target classes, for all samples z'
                Dimension (N * C) where N is num_samples and C num_classses. 
        """
        # Create a networkx graph 
        G = custom_to_networkx(self.data)
        G = G.subgraph(self.neighbours.tolist() + [node_index])

        # Define an "average" feature vector - for discarded features
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.data.x.mean(dim=0)
            # Change here for contrastive explanations
            # av_feat_values = self.data.x[402]
            # or random feature vector made of random value across each col of X

        # Store discarded nodes/features (z_j=0) for each sample z 
        excluded_feat = {}
        excluded_nei = {}
        for i in range(num_samples):

            # Excluded features' indexes 
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Excluded neighbors' indexes 
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init label f(z') for graphshap dataset
        if multiclass:
            # Allows to explain why one class was not chosen 
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Construct new matrices A and X for each sample - reform z' from z
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            # Isolate in the graph each node excluded from the sampled coalition
            positions = []
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)
 
            # Set features not in the sampled coalition to an average value
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]

            # Discared features approximation
            # if args_feat != 'Null' and discarded_feat_idx!=[] and D - len(ex_nei) < args_K:
            #     X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Indirect effect - if few included neighbours
            # Make sure that they are connected to v (with current nodes sampled nodes)
            if 0 < D - len(ex_nei) < args_K:
                included_nei = set(
                    self.neighbours.detach().numpy()).difference(ex_nei)
                included_nei = included_nei.difference(
                    one_hop_neighbours.detach().numpy())
                for incl_nei in included_nei:
                    paths = list(nx.all_shortest_paths(G, source=node_index, target=incl_nei))
                    np.random.shuffle(paths)
                    len_paths = [len(set(path[1:-1]).intersection(ex_nei)) for path in paths]
                    if min(len_paths) == 0: 
                        pass
                    else: 
                        path = paths[np.argmin(len_paths)]
                        for n in range(1, len(path)-1):
                            A = torch.cat((A, torch.tensor(
                                            [[path[n-1]], [path[n]]])), dim=-1)
                            X[path[n], :] = X[node_index, :]  # av_feat_values
                            # TODO: eval this against av.values.

            # Apply model on new (X,A) 
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(X.cuda(), A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model(X, A).exp()[node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]
                
        return fz

    def graph_classification(self, graph_index, num_samples, D, z_, args_K, args_feat, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Graph Classification task

        Args:
            Variables are defined exactly as defined in explainer function
            Note that adjacency matrices are dense (square) matrices (unlike node classification)

        Returns:
            (tensor): f(z') - probability of belonging to each target classes, for all samples z'
            Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Store discarded nodes (z_j=0) for each sample z 
        excluded_nei = {}
        for i in range(num_samples):
            # Excluded nodes' indexes 
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j])
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init 
        fz = torch.zeros(num_samples)
        adj = deepcopy(self.data.edge_index[graph_index])
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x[graph_index].shape[1])
        else: 
            av_feat_values = self.data.x.mean(dim=0).mean(dim=0)
            #av_feat_values = np.mean(self.data.x[graph_index],axis=0)
        
        # Create new matrix A and X - for each sample ≈ reform z' from z
        for (key, ex_nei) in tqdm(excluded_nei.items()):

            # Change adj matrix
            A = deepcopy(adj)
            A[ex_nei, :] = 0
            A[:, ex_nei] = 0

            # Also change features of excluded nodes (optional)
            X = deepcopy(self.data.x[graph_index])
            for nei in ex_nei:
                X[nei] = av_feat_values

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(X.unsqueeze(0).cuda(), A.unsqueeze(0).cuda()).exp()
            else:
                with torch.no_grad():
                    proba = self.model(X.unsqueeze(0), A.unsqueeze(0)).exp()

            # Compute prediction
            fz[key] = proba[0][true_pred.item()]

        return fz

    def basic_default(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Does not deal with isolated 2 hops neighbours (or more)

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z') - probability of belonging to each target classes, for all samples z'
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Define an "average" feature vector - for discarded features
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.data.x.mean(dim=0)

        # Define excluded_feat and excluded_nei for each z
        excluded_feat = {}
        excluded_nei = {}
        for i in range(num_samples):

            # Excluded features' indexes 
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Excluded neighbors' indexes 
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Construct new matrices A and X for each sample - reform z from z'
        for (key, ex_nei), (_, ex_feat) in tqdm(zip(excluded_nei.items(), excluded_feat.items())):

            # For each excluded neighbour, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            positions = []
            for val in ex_nei:
                pos = (self.data.edge_index == val).nonzero()[:, 1].tolist()
                positions += pos
            positions = list(set(positions))
            A = np.array(self.data.edge_index)
            A = np.delete(A, positions, axis=1)
            A = torch.tensor(A)

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)
            X[node_index, ex_feat] = av_feat_values[ex_feat]

            # Discarded features approx
            # if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
            #     X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

             # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(X.cuda(), A.cuda()).exp()[node_index]
            else:
                with torch.no_grad():
                    proba = self.model(X, A).exp()[node_index]
                
            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else:
                fz[key] = proba[true_pred]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else: 
                fz[key] = proba[true_pred]

        return fz

    def neutral(self, node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat, discarded_feat_idx, multiclass, true_pred):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Do not isolate nodes but set their feature vector to expected values
            Consider node features for node itself

        Args:
                Variables are defined exactly as defined in explainer function

        Returns:
                (tensor): f(z') - probability of belonging to each target classes, for all samples z
                Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Initialise new node feature vectors and neighbours to disregard
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.num_features)
        else:
            av_feat_values = self.data.x.mean(dim=0)
        # or random feature vector made of random value across each col of X

        excluded_feat = {}
        excluded_nei = {}

        # Define excluded_feat and excluded_nei for each z'
        for i in tqdm(range(num_samples)):

            # Define new node features dataset (we only modify x_v for now)
            # Store index of features that are not sampled (z_j=0)
            feats_id = []
            for j in range(self.F):
                if z_[i, j].item() == 0:
                    feats_id.append(feat_idx[j].item())
            excluded_feat[i] = feats_id

            # Define new neighbourhood
            # Store index of neighbours that need to be isolated (not sampled, z_j=0)
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j].item())
            # Dico with key = num_sample id, value = excluded neighbour index
            excluded_nei[i] = nodes_id

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Create new matrix A and X - for each sample ≈ reform z' from z
        for (key, ex_nei), (_, ex_feat) in zip(excluded_nei.items(), excluded_feat.items()):

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)

            # For each excluded node, retrieve the column index of its occurences
            # in the adj matrix - store them in positions (list)
            A = self.data.edge_index
            X[ex_nei, :] = av_feat_values.repeat(len(ex_nei), 1)
            # Set all excluded features to expected value for node index only
            X[node_index, ex_feat] = av_feat_values[ex_feat]
            if args_feat != 'Null' and discarded_feat_idx != [] and len(self.neighbours) - len(ex_nei) < args_K:
                X[node_index, discarded_feat_idx] = av_feat_values[discarded_feat_idx]

            # Apply model on (X,A) as input.
            if self.gpu:
                with torch.no_grad():
                    proba = self.model(X.cuda(), self.data.edge_index.cuda()).exp()[
                        node_index]
            else: 
                with torch.no_grad():
                    proba = self.model(X, self.data.edge_index).exp()[
                        node_index]

            # Store predicted class label in fz
            if multiclass:
                fz[key] = proba
            else: 
                fz[key] = proba[true_pred]

        return fz

        ################################

    ################################
    # Explanation Generator
    ################################

    def WLS(self, z_, weights, fz, multiclass, info):
        """ Weighted Least Squares Method
            Estimates shapley values via explanation model

        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z
            fz ([type]): prediction f(z') where z' is a new instance - formed from z and x

        Returns:
            [tensor]: estimated coefficients of our weighted linear regression - on (z, f(z'))
            Dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            if info:
                print('WLS: Matrix not invertible')
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(10**(-5) * np.random.randn(tmp.shape[1])))

        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))

        # Test accuracy
        y_pred = z_.detach().numpy() @ phi
        if info:
            print('r2: ', r2_score(fz, y_pred))
            print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]

    def WLR_sklearn(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression

        Args:
            z_ (torch.tensor): dataset
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): predictions for z_ 

        Return:
            tensor: parameters of explanation model g
        """
        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()

        # Fit weighted linear regression
        reg = LinearRegression()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)

        # Assess perf
        if info:
            print('weighted r2: ', reg.score(z_, fz, sample_weight=weights))
            print('r2: ', r2_score(fz, y_pred))

        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value

    def WLR_Lasso(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression with lasso regularisation

        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data 
        
        Return:
            tensor: parameters of explanation model g
        
        """
        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()
        # Fit weighted linear regression
        reg = Lasso(alpha=0.01)
        # reg = Lasso()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)
        # Assess perf
        if info:
            print('weighted r2: ', reg.score(z_, fz, sample_weight=weights))
            print('r2: ', r2_score(fz, y_pred))
        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value

    def WLR(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression

        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data 
        
        Return:
            tensor: parameters of explanation model g
        """
        # Define model
        if multiclass:
            our_model = LinearRegressionModel(
                z_.shape[1], self.data.num_classes)
        else:
            our_model = LinearRegressionModel(z_.shape[1], 1)
        our_model.train()

        # Define optimizer and loss function
        def weighted_mse_loss(input, target, weight):
            return (weight * (input - target) ** 2).mean()

        criterion = torch.nn.MSELoss()
        #optimizer = torch.optim.SGD(our_model.parameters(), lr=0.2)
        optimizer = torch.optim.Adam(our_model.parameters(), lr=0.001)

        # Dataloader
        train = torch.utils.data.TensorDataset(z_, fz, weights)
        train_loader = torch.utils.data.DataLoader(train, batch_size=1)

        # Repeat for several epochs
        for epoch in range(100):

            av_loss = []
            #for x,y,w in zip(z_,fz, weights):
            for batch_idx, (dat, target, w) in enumerate(train_loader):
                x, y, w = Variable(dat), Variable(target), Variable(w)

                # Forward pass: Compute predicted y by passing x to the model
                pred_y = our_model(x)

                # Compute loss
                loss = weighted_mse_loss(pred_y, y, w)
                #loss = criterion(pred_y,y)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store batch loss
                av_loss.append(loss.item())
            if epoch % 10 ==0 and info:
                print('av loss epoch: ', np.mean(av_loss))

        # Evaluate model
        our_model.eval()
        with torch.no_grad():
            pred = our_model(z_)
        if info:
            print('weighted r2 score: ', r2_score(
                pred, fz, multioutput='variance_weighted'))
            if multiclass:
                print(r2_score(pred, fz, multioutput='raw_values'))
            print('r2 score: ', r2_score(pred, fz, weights))

        phi, base_value = [param.T for _,
                           param in our_model.named_parameters()]
        phi = np.squeeze(phi, axis=1)
        return phi.detach().numpy().astype('float64'), base_value.detach().numpy().astype('float64')
 

    ################################
    # INFO ON EXPLANATIONS
    ################################

    def print_info(self, D, node_index, phi, feat_idx, true_pred, true_conf, base_value, multiclass):
        """
        Displays some information about explanations - for a better comprehension and audit
        """
        # Print some information
        print('Explanations include {} node features and {} neighbours for this node\
        for {} classes'.format(self.F, D, self.data.num_classes))

        # Compare with true prediction of the model - see what class should truly be explained
        print('Model prediction is class {} with confidence {}, while true label is {}'
              .format(true_pred, true_conf, self.data.y[node_index]))

        # Print base value
        print('Base value', base_value, 'for class ', true_pred.item())

        # Isolate explanations for predicted class - explain model choices
        if multiclass:
            pred_explanation = phi[true_pred, :]
        else:
            pred_explanation = phi

        # print('Explanation for the class predicted by the model:', pred_explanation)

        # Look at repartition of weights among neighbours and node features
        # Motivation for regularisation
        print('Weights for node features: ', sum(pred_explanation[:self.F]),
              'and neighbours: ', sum(pred_explanation[self.F:]))
        #print('Total Weights (abs val) for node features: ', sum(np.abs(pred_explanation[:self.F])),
        #      'and neighbours: ', sum(np.abs(pred_explanation[self.F:])))

        # Proportional importance granted to graph structure vs node features of v
        #print('Feature importance wrt explainable part: {} %'.format( 100 * sum(pred_explanation[:self.F]) / (true_pred.item())))
        #print('Node importance wrt explainable part: {} %'.format(100* sum(pred_explanation[self.F:]) / (true_pred.item())) )

        # Note we focus on explanation for class predicted by the model here, so there is a bias towards
        # positive weights in our explanations (proba is close to 1 everytime).
        # Alternative is to view a class at random or the second best class

        # Select most influential neighbours and/or features (+ or -)
        if self.F + D > 10:
            _, idxs = torch.topk(torch.from_numpy(np.abs(pred_explanation)), 6)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                if idx.item() < self.F:
                    influential_feat[feat_idx[idx]] = val
                else:
                    influential_nei[self.neighbours[idx-self.F]] = val
            print('Most influential features: ', len([(item[0].item(), item[1].item()) for item in list(influential_feat.items())]),
                  'and neighbours', len([(item[0].item(), item[1].item()) for item in list(influential_nei.items())]))

        # Most influential features splitted bewteen neighbours and features
        if self.F > 5:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[:self.F])), 3)
            vals = [pred_explanation[idx] for idx in idxs]
            influential_feat = {}
            for idx, val in zip(idxs, vals):
                influential_feat[feat_idx[idx]] = val
            print('Most influential features: ', [
                  (item[0].item(), item[1].item()) for item in list(influential_feat.items())])

        # Most influential features splitted bewteen neighbours and features
        if D > 5 and self.M != self.F:
            _, idxs = torch.topk(torch.from_numpy(
                np.abs(pred_explanation[self.F:])), 3)
            vals = [pred_explanation[self.F + idx] for idx in idxs]
            influential_nei = {}
            for idx, val in zip(idxs, vals):
                influential_nei[self.neighbours[idx]] = val
            print('Most influential neighbours: ', [
                  (item[0].item(), item[1].item()) for item in list(influential_nei.items())])

    def vizu(self, edge_mask, node_index, phi, predicted_class, hops, multiclass):
        """ Vizu of important nodes in subgraph around node_index

        Args:
            edge_mask ([type]): vector of size data.edge_index with False 
                                            if edge is not included in subgraph around node_index
            node_index ([type]): node of interest index
            phi ([type]): explanations for node of interest
            predicted_class ([type]): class predicted by model for node of interest 
            hops ([type]):  number of hops considered for subgraph around node of interest 
            multiclass: if we look at explanations for all classes or only for the predicted one
        """
        if multiclass:
            phi = torch.tensor(phi[predicted_class, :])
        else:
            phi = torch.from_numpy(phi).float()

        # Replace False by 0, True by 1 in edge_mask
        mask = edge_mask.int().float()

        # Identify one-hop subgraph around node_index
        one_hop_nei, _, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_index, 1, self.data.edge_index, relabel_nodes=True,
            num_nodes=None)
        #true_one_hop_nei = one_hop_nei[one_hop_nei != node_index]

        # Attribute phi to edges in subgraph bsed on the incident node phi value
        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
            for idx in list_indexes:
                # Remove importance of 1-hop neighbours to 2-hop nei.
                if nei in one_hop_nei:
                    if self.data.edge_index[1, idx] in one_hop_nei:
                        mask[idx] = phi[self.F + i]
                elif mask[idx] == 1:
                    mask[idx] = phi[self.F + i]
            #mask[mask.nonzero()[i].item()]=phi[i, predicted_class]

        # Set to 0 importance of edges related to node_index
        mask[mask == 1] = 0

        # Increase coef for visibility and consider absolute contribution
        mask = torch.abs(mask)
        mask = mask / sum(mask)

        # Vizu nodes
        ax, G = visualize_subgraph(self.model,
                                   node_index,
                                   self.data.edge_index,
                                   mask,
                                   hops,
                                   y=self.data.y,
                                   threshold=None)

        plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
                                                  self.model.__class__.__name__,
                                                  node_index),
                    bbox_inches='tight')

        # Other visualisation
        G = denoise_graph(self.data, mask, phi[self.F:], self.neighbours,
                  node_index, feat=None, label=self.data.y, threshold_num=10)

        log_graph(G,
                  identify_self=True,
                  nodecolor="label",
                  epoch=0,
                  fig_size=(4, 3),
                  dpi=300,
                  label_node_feat=False,
                  edge_vmax=None,
                  args=None)

        plt.savefig('results/GS_{}_{}_{}'.format(self.data.name,
                                                 self.model.__class__.__name__,
                                                 node_index),
                    bbox_inches='tight')

        #plt.show()


################################
# SOME BASELINES
################################

class Greedy:

    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.neighbours = None
        self.gpu = gpu
        self.M = self.data.num_features
        self.F = self.M

        self.model.eval()

    def explain(self, node_index=0, hops=2, num_samples=0, info=False, multiclass=False, *unused):
        """
        Greedy explainer - only considers node features for explanations
        Computes the prediction proba with and without the targeted feature (repeat for all feat)
        This feature's importance is set as the normalised absolute difference in predictions above
        :param num_samples, info: useless here (simply to match GraphSVX structure)
        """
        # Store indexes of these non zero feature values
        feat_idx = torch.arange(self.F)

        # Compute predictions
        if self.gpu:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            with torch.no_grad():
                probas = self.model(x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                    node_index]
        else:
            with torch.no_grad():
                probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
                    node_index]
        probas, label = torch.topk(probas, k=1)

        # Init explanations vector

        if multiclass:
            coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)

            # Loop on all features - consider all classes
            for i, idx in enumerate(feat_idx):
                idx = idx.item()
                x_ = deepcopy(self.data.x)
                x_[:, idx] = 0.0  # set feat of interest to 0
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=x_.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index]  # [label].item()
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[
                            node_index]  # [label].item()
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_)/probas).detach().numpy()
        else:
            #probas = probas[label.item()]
            coefs = np.zeros([self.M])  # (m, #feats)
            # Loop on all features - consider all classes
            for i, idx in enumerate(feat_idx):
                idx = idx.item()
                x_ = deepcopy(self.data.x)
                x_[:, idx] = 0.0  # set feat of interest to 0
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=x_.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index, label.item()]  # [label].item()
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[
                            node_index, label.item()]  # [label].item()
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_) /
                            probas).cpu().detach().numpy()

        return coefs

    def explain_nei(self, node_index=0, hops=2, num_samples=0, info=False, multiclass=False):
        """Greedy explainer - only considers node features for explanations
        """
        # Construct k hop subgraph of node of interest (denoted v)
        neighbours, _, _, edge_mask =\
            torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                 num_hops=hops,
                                                 edge_index=self.data.edge_index)
        # Store the indexes of the neighbours of v (+ index of v itself)

        # Remove node v index from neighbours and store their number in D
        neighbours = neighbours[neighbours != node_index]
        self.neighbours = neighbours
        self.M = neighbours.shape[0]

        # Compute predictions
        if self.gpu:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            with torch.no_grad():
                probas = self.model(x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                    node_index]
        else:
            with torch.no_grad():
                probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
                    node_index]
        pred_confidence, label = torch.topk(probas, k=1)

        if multiclass:
            # Init explanations vector
            coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)

            # Loop on all neighbours - consider all classes
            for i, nei_idx in enumerate(self.neighbours):
                nei_idx = nei_idx.item()
                A_ = deepcopy(self.data.edge_index)

                # Find all edges incident to the isolated neighbour
                pos = (self.data.edge_index == nei_idx).nonzero()[
                    :, 1].tolist()

                # Create new adjacency matrix where this neighbour is isolated
                A_ = np.array(self.data.edge_index)
                A_ = np.delete(A_, pos, axis=1)
                A_ = torch.tensor(A_)

                # Compute new prediction with updated adj matrix (without this neighbour)
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x.cuda(), edge_index=A_.cuda()).exp()[
                            node_index]
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x, edge_index=A_).exp()[
                            node_index]
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_) /
                            probas).cpu().detach().numpy()

        else:
            probas = probas[label.item()]
            coefs = np.zeros(self.M)
            for i, nei_idx in enumerate(self.neighbours):
                nei_idx = nei_idx.item()
                A_ = deepcopy(self.data.edge_index)

                # Find all edges incident to the isolated neighbour
                pos = (self.data.edge_index == nei_idx).nonzero()[
                    :, 1].tolist()

                # Create new adjacency matrix where this neighbour is isolated
                A_ = np.array(self.data.edge_index)
                A_ = np.delete(A_, pos, axis=1)
                A_ = torch.tensor(A_)

                # Compute new prediction with updated adj matrix (without this neighbour)
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x.cuda(), edge_index=A_.cuda()).exp()[
                            node_index, label.item()]
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x, edge_index=A_).exp()[
                            node_index, label.item()]
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_) /
                            probas).cpu().detach().numpy()

        return coefs


class Random:
    """Random explainer - selects random variables as explanations
    """

    def __init__(self, num_feats, K):
        self.num_feats = num_feats
        self.K = K

    def explain(self):
        return np.random.choice(self.num_feats, self.K)


class GraphLIME:
    """ GraphLIME explainer - code taken from original repository
    Explains only node features

    """

    def __init__(self, data, model, gpu=False, hop=2, rho=0.1, cached=True):
        self.data = data
        self.model = model
        self.hop = hop
        self.rho = rho
        self.cached = cached
        self.cached_result = None
        self.M = self.data.num_features
        self.F = self.data.num_features
        self.gpu = gpu

        self.model.eval()

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, y, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hop, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        y = y[subset]

        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, y, edge_index, mapping, edge_mask, kwargs

    def __init_predict__(self, x, edge_index, **kwargs):
        if self.cached and self.cached_result is not None:
            if x.size(0) != self.cached_result.size(0):
                raise RuntimeError(
                    'Cached {} number of nodes, but found {}.'.format(
                        x.size(0), self.cached_result.size(0)))

        if not self.cached or self.cached_result is None:
            # Get the initial prediction.
            with torch.no_grad():
                if self.gpu:
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    self.model = self.model.to(device)
                    log_logits = self.model(
                        x=x.cuda(), edge_index=edge_index.cuda(), **kwargs)
                else:
                    log_logits = self.model(
                        x=x, edge_index=edge_index, **kwargs)
                probas = log_logits.exp()

            self.cached_result = probas

        return self.cached_result

    def __compute_kernel__(self, x, reduce):
        assert x.ndim == 2, x.shape

        n, d = x.shape

        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # (n, n, d)
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)  # (n, n, 1)

        std = np.sqrt(d)

        # (n, n, 1) or (n, n, d)
        K = np.exp(-dist / (2 * std ** 2 * 0.1 + 1e-10))

        return K

    def __compute_gram_matrix__(self, x):
        # unstable implementation due to matrix product (HxH)
        # n = x.shape[0]
        # H = np.eye(n, dtype=np.float) - 1.0 / n * np.ones(n, dtype=np.float)
        # G = np.dot(np.dot(H, x), H)

        # more stable and accurate implementation
        G = x - np.mean(x, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)

        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)

        return G

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, *unused, **kwargs):
        # hops, num_samples, info are useless: just to copy graphshap pipeline
        x = self.data.x
        edge_index = self.data.edge_index

        probas = self.__init_predict__(x, edge_index, **kwargs)

        x, probas, _, _, _, _ = self.__subgraph__(
            node_index, x, probas, edge_index, **kwargs)

        x = x.cpu().detach().numpy()  # (n, d)
        y = probas.cpu().detach().numpy()  # (n, classes)

        n, d = x.shape

        if multiclass:
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=False)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2, self.data.num_classes)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False,
                               normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_.T

        else:
            K = self.__compute_kernel__(x, reduce=False)  # (n, n, d)
            L = self.__compute_kernel__(y, reduce=True)  # (n, n, 1)

            K_bar = self.__compute_gram_matrix__(K)  # (n, n, d)
            L_bar = self.__compute_gram_matrix__(L)  # (n, n, 1)

            K_bar = K_bar.reshape(n ** 2, d)  # (n ** 2, d)
            L_bar = L_bar.reshape(n ** 2,)  # (n ** 2,)

            solver = LassoLars(self.rho, fit_intercept=False,
                               normalize=False, positive=True)
            solver.fit(K_bar * n, L_bar * n)

            return solver.coef_


class LIME:
    """ LIME explainer - adapted to GNNs
    Explains only node features

    """

    def __init__(self, data, model, gpu=False, cached=True):
        self.data = data
        self.model = model
        self.gpu = gpu
        self.M = self.data.num_features
        self.F = self.data.num_features

        self.model.eval()

    def __init_predict__(self, x, edge_index, *unused, **kwargs):

        # Get the initial prediction.
        with torch.no_grad():
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                log_logits = self.model(
                    x=x.cuda(), edge_index=edge_index.cuda(), **kwargs)
            else:
                log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            probas = log_logits.exp()

        return probas

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, **kwargs):
        num_samples = num_samples//2
        x = self.data.x
        edge_index = self.data.edge_index

        probas = self.__init_predict__(x, edge_index, **kwargs)
        proba, label = probas[node_index, :].max(dim=0)

        x_ = deepcopy(x)
        original_feats = x[node_index, :]

        if multiclass:
            sample_x = [original_feats.detach().numpy()]
            #sample_y = [proba.item()]
            sample_y = [probas[node_index, :].detach().numpy()]

            for _ in range(num_samples):
                x_[node_index, :] = original_feats + \
                    torch.randn_like(original_feats)

                with torch.no_grad():
                    if self.gpu:
                        log_logits = self.model(
                            x=x_.cuda(), edge_index=edge_index.cuda(), **kwargs)
                    else:
                        log_logits = self.model(
                            x=x_, edge_index=edge_index, **kwargs)
                    probas_ = log_logits.exp()

                #proba_ = probas_[node_index, label]
                proba_ = probas_[node_index]

                sample_x.append(x_[node_index, :].detach().numpy())
                # sample_y.append(proba_.item())
                sample_y.append(proba_.detach().numpy())

        else:
            sample_x = [original_feats.detach().numpy()]
            sample_y = [proba.item()]
            # sample_y = [probas[node_index, :].detach().numpy()]

            for _ in range(num_samples):
                x_[node_index, :] = original_feats + \
                    torch.randn_like(original_feats)

                with torch.no_grad():
                    if self.gpu:
                        log_logits = self.model(
                            x=x_.cuda(), edge_index=edge_index.cuda(), **kwargs)
                    else:
                        log_logits = self.model(
                            x=x_, edge_index=edge_index, **kwargs)
                    probas_ = log_logits.exp()

                proba_ = probas_[node_index, label]
                # proba_ = probas_[node_index]

                sample_x.append(x_[node_index, :].detach().numpy())
                sample_y.append(proba_.item())
                # sample_y.append(proba_.detach().numpy())

        sample_x = np.array(sample_x)
        sample_y = np.array(sample_y)

        solver = Ridge(alpha=0.1)
        solver.fit(sample_x, sample_y)

        return solver.coef_.T


class SHAP():
    """ KernelSHAP explainer - adapted to GNNs
    Explains only node features

    """
    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        # number of nonzero features - for each node index
        self.M = self.data.num_features
        self.neighbours = None
        self.F = self.M

        self.model.eval()

    def explain(self, node_index=0, hops=2, num_samples=10, info=True, multiclass=False, *unused):
        """
        :param node_index: index of the node of interest
        :param hops: number k of k-hop neighbours to consider in the subgraph around node_index
        :param num_samples: number of samples we want to form GraphSVX's new dataset 

        :return: shapley values for features that influence node v's pred
        """
        # Compute true prediction of model, for original instance
        with torch.no_grad():
            if self.gpu:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.model = self.model.to(device)
                true_conf, true_pred = self.model(
                    x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[node_index][0].max(dim=0)
            else:
                true_conf, true_pred = self.model(
                    x=self.data.x, edge_index=self.data.edge_index).exp()[node_index][0].max(dim=0)

        # Determine z => features whose importance is investigated
        # Decrease number of samples because nodes are not considered
        num_samples = num_samples//3

        # Consider all features (+ use expectation like below)
        # feat_idx = torch.unsqueeze(torch.arange(self.F), 1)

        # Sample z - binary vector of dimension (num_samples, M)
        z_ = torch.empty(num_samples, self.M).random_(2)
        # Compute |z| for each sample z
        s = (z_ != 0).sum(dim=1)

        # Define weights associated with each sample using shapley kernel formula
        weights = self.shapley_kernel(s)

        # Create dataset (z, f(z')), stored as (z_, fz)
        # Retrive z' from z and x_v, then compute f(z')
        fz = self.compute_pred(node_index, num_samples,
                            z_, multiclass, true_pred)

        # OLS estimator for weighted linear regression
        phi, base_value = self.OLS(z_, weights, fz)  # dim (M*num_classes)

        return phi

    def shapley_kernel(self, s):
        """
        :param s: dimension of z' (number of features + neighbours included)
        :return: [scalar] value of shapley value 
        """
        shap_kernel = []
        # Loop around elements of s in order to specify a special case
        # Otherwise could have procedeed with tensor s direclty
        for i in range(s.shape[0]):
            a = s[i].item()
            # Put an emphasis on samples where all or none features are included
            if a == 0 or a == self.M:
                shap_kernel.append(1000)
            elif scipy.special.binom(self.M, a) == float('+inf'):
                shap_kernel.append(1/self.M)
            else:
                shap_kernel.append(
                    (self.M-1)/(scipy.special.binom(self.M, a)*a*(self.M-a)))
        return torch.tensor(shap_kernel)

    def compute_pred(self, node_index, num_samples, z_, multiclass, true_pred):
        """
        Variables are exactly as defined in explainer function, where compute_pred is used
        This function aims to construct z' (from z and x_v) and then to compute f(z'), 
        meaning the prediction of the new instances with our original model. 
        In fact, it builds the dataset (z, f(z')), required to train the weighted linear model.

        :return fz: probability of belonging to each target classes, for all samples z'
        fz is of dimension N*C where N is num_samples and C num_classses. 
        """
        # This implies retrieving z from z' - wrt sampled neighbours and node features
        # We start this process here by storing new node features for v and neigbours to
        # isolate
        X_v = torch.zeros([num_samples, self.F])

        # Init label f(z') for graphshap dataset - consider all classes
        if multiclass:
            fz = torch.zeros((num_samples, self.data.num_classes))
        else:
            fz = torch.zeros(num_samples)

        # Do it for each sample
        for i in range(num_samples):

            # Define new node features dataset (we only modify x_v for now)
            # Features where z_j == 1 are kept, others are set to 0
            for j in range(self.F):
                if z_[i, j].item() == 1:
                    X_v[i, j] = 1

            # Change feature vector for node of interest
            X = deepcopy(self.data.x)
            X[node_index, :] = X_v[i, :]

            # Apply model on (X,A) as input.
            with torch.no_grad():
                if self.gpu:
                    proba = self.model(x=X.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                        node_index]
                else:
                    proba = self.model(x=X, edge_index=self.data.edge_index).exp()[
                        node_index]
            # Multiclass
            if not multiclass:
                fz[i] = proba[true_pred]
            else:
                fz[i] = proba

        return fz

    def OLS(self, z_, weights, fz):
        """
        :param z_: z - binary vector  
        :param weights: shapley kernel weights for z
        :param fz: f(z') where z is a new instance - formed from z and x
        :return: estimated coefficients of our weighted linear regression - on (z, f(z'))
        phi is of dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(0.00001 * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(
            np.dot(z_.T, np.diag(weights)), fz.cpu().detach().numpy()))

        # Test accuracy
        # y_pred=z_.detach().numpy() @ phi
        #	print('r2: ', r2_score(fz, y_pred))
        #	print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]


class GNNExplainer():
    """GNNExplainer - use python package 
    Specific to GNNs: explain node features and graph structure

    """

    def __init__(self, data, model, gpu=False):
        self.data = data
        self.model = model
        self.M = self.data.num_nodes + self.data.num_features
        self.gpu = gpu
        # self.coefs = torch.zeros(self.data.num_nodes, self.data.num_classes)
        self.coefs = None  # node importance derived from edge importance
        self.edge_mask = None
        self.neighbours = None
        self.F = self.data.num_features

        self.model.eval()

    def explain(self, node_index, hops, num_samples, info=False, multiclass=False, *unused):
        num_samples = num_samples//2
        # Use GNNE open source implem - outputs features's and edges importance
        if self.gpu:
            device = torch.device('cpu')
            self.model = self.model.to(device)
        explainer = GNNE(self.model, epochs=150)
        try:
            node_feat_mask, self.edge_mask = explainer.explain_node(
                node_index, self.data.x, self.data.edge_index)
        except AssertionError:
            node_feat_mask, self.edge_mask = explainer.explain_node(
                np.random.choice(range(1000)), self.data.x, self.data.edge_index)

        # Transfer edge importance to node importance
        # Node importance = average of incident edges importance
        dico = {}
        for idx in torch.nonzero(self.edge_mask):
            node = self.data.edge_index[0, idx].item()
            if not node in dico.keys():
                dico[node] = [self.edge_mask[idx]]
            else:
                dico[node].append(self.edge_mask[idx])
        # Count neighbours in the subgraph
        self.neighbours = torch.tensor([index for index in dico.keys()])

        if multiclass:
            # Attribute an importance measure to each node = sum of incident edges' importance
            self.coefs = torch.zeros(
                self.neighbours.shape[0], self.data.num_classes)
            # for key, val in dico.items():
            for i, val in enumerate(dico.values()):
                # self.coefs[i, :] = sum(val)
                self.coefs[i, :] = max(val)
                

            # Eliminate node_index from neighbourhood
            self.neighbours = self.neighbours[self.neighbours != node_index]
            self.coefs = self.coefs[1:]

            if info == True:
                self.vizu(self.edge_mask, node_index, self.coefs[0], hops)

            return torch.stack([node_feat_mask]*self.data.num_classes, 1)

        else:
            # Attribute an importance measure to each node = sum of incident edges' importance
            self.coefs = torch.zeros(
                self.neighbours.shape[0])
            for i, val in enumerate(dico.values()):
                self.coefs[i] = sum(val)

            # Eliminate node_index from neighbourhood
            j = (self.neighbours == node_index).nonzero().item()
            self.coefs = torch.cat([self.coefs[:j], self.coefs[j+1:]])
            self.neighbours = self.neighbours[self.neighbours != node_index]
            
            if info == True:
                self.vizu(self.edge_mask, node_index, self.coefs[0], hops)
            del explainer

            return node_feat_mask

    def vizu(self, edge_mask, node_index, phi, hops):
        """
        Visualize the importance of neighbours in the subgraph of node_index
        """
        # Replace False by 0, True by 1
        mask = torch.zeros(self.data.edge_index.shape[1])
        for i, val in enumerate(edge_mask):
            if val.item() != 0:
                mask[i] = 1

        # Attribute phi to edges in subgraph bsed on the incident node phi value
        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
            for idx in list_indexes:
                if self.data.edge_index[1, idx] == node_index:
                    mask[idx] = edge_mask[idx]
                    break
                elif mask[idx] != 0:
                    mask[idx] = edge_mask[idx]

        # Set to 0 importance of edges related to 0
        mask[mask == 1] = 0

        # Vizu nodes and
        ax, G = visualize_subgraph(self.model,
                                   node_index,
                                   self.data.edge_index,
                                   mask,
                                   hops,
                                   y=self.data.y,
                                   threshold=None)

        plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name,
                                                  self.model.__class__.__name__,
                                                  node_index),
                    bbox_inches='tight')
        #plt.show()
