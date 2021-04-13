"""graph_utils.py

   Utility for sampling graphs from a dataset.
"""
import sys
from scipy.sparse import coo_matrix
import pickle as pkl
from configs import *
import networkx as nx
import numpy as np
import torch
import torch.utils.data


class GraphSampler(torch.utils.data.Dataset):
    """ Create proper dataset format that DataLoader will understand
    """

    def __init__(self,data):

        # Train
        self.train_adj = [row.squeeze() for row in data.edge_index[(
            data.train_mask == True).nonzero()]]
        self.train_feat = [row.squeeze() for row in data.x[(data.train_mask==True).nonzero()]]
        self.train_label = data.y[(data.train_mask == True).nonzero().view(-1)].tolist()

        # # Val
        # self.val_adj = [row for row in data.edge_index[(
        #     data.val_mask == True).nonzero()]]
        # self.val_feat = [row for row in data.x[(
        #     data.val_mask == True).nonzero()]]
        # self.val_label = data.y[(data.val_mask == True).nonzero()].tolist()

    def __len__(self):
        return len(self.train_adj)

    def __getitem__(self, idx):
        adj = self.train_adj[idx]
        num_nodes = adj.shape[0]

        return {
            "adj": adj,
            "feats": self.train_feat[idx],
            "label": self.train_label[idx],
            "num_nodes": num_nodes,
        }


def get_graph_data(dataset):
    pri = './data/'+dataset+'/'+dataset+'_'

    file_edges = pri+'A.txt'
    file_edge_labels = pri+'edge_labels.txt'
    file_edge_labels = pri+'edge_gt.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'
    file_node_labels = pri+'node_labels.txt'

    edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
    try:
        edge_labels = np.loadtxt(
            file_edge_labels, delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use edge label 0')
        edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

    graph_indicator = np.loadtxt(
        file_graph_indicator, delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(
        file_graph_labels, delimiter=',').astype(np.int32)

    try:
        node_labels = np.loadtxt(
            file_node_labels, delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use node label 0')
        node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

    graph_id = 1
    starts = [1]
    node2graph = {}
    for i in range(len(graph_indicator)):
        if graph_indicator[i] != graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1] = len(starts)-1
    # print(starts)
    # print(node2graph)
    graphid = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    for (s, t), l in list(zip(edges, edge_labels)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid != tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s, t, 'graph id', sgid, tgid)
            exit(1)
        gid = sgid
        if gid != graphid:
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_list = []
            edge_label_list = []
            graphid = gid
        start = starts[gid]
        edge_list.append((s-start, t-start))
        edge_label_list.append(l)

    edge_lists.append(edge_list)
    edge_label_lists.append(edge_label_list)

    # node labels
    node_label_lists = []
    graphid = 0
    node_label_list = []
    for i in range(len(node_labels)):
        nid = i+1
        gid = node2graph[nid]
        # start = starts[gid]
        if gid != graphid:
            node_label_lists.append(node_label_list)
            graphid = gid
            node_label_list = []
        node_label_list.append(node_labels[i])
    node_label_lists.append(node_label_list)

    return edge_lists, graph_labels, edge_label_lists, node_label_lists
