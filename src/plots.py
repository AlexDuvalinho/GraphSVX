import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from copy import copy
from math import sqrt
import statistics
import torch
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx, subgraph


def plot_dist(noise_feats, label=None, ymax=1.1, color=None, title=None, save_path=None):
    """
    Kernel density plot of the number of noisy features included in explanations, 
    for a certain number of test samples
    """
    if not any(noise_feats):  # handle special case where noise_feats=0
        noise_feats[0] = 0.25

    # plt.switch_backend("agg")
    sns.set_style('darkgrid')
    #sns.set_context("talk")
    ax = sns.distplot(noise_feats, hist=False, kde=True,
                      kde_kws={'label': label}, color=color)
    sns.set(font_scale=1.5)  # , rc={"lines.linewidth": 2})
    plt.xlim(-3, 8)
    plt.ylim(ymin=0.0, ymax=ymax)

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)

    return ax


def __flow__(model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            return module.flow
    return 'source_to_target'


def visualize_subgraph(model, node_idx, edge_index, edge_mask, num_hops, y=None,
                       threshold=None, **kwargs):
    """Visualizes the subgraph around :attr:`node_idx` given an edge mask
    :attr:`edge_mask`.

    Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices - adj matrix 
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                    as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                    important edges. If set to :obj:`None`, will visualize all
                    edges with transparancy indicating the importance of edges.
                    (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                    :func:`nx.draw`.

    :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
    """
    assert edge_mask.size(0) == edge_index.size(1)
    edge_index = edge_index[:, edge_mask.nonzero().T[0]]

    # Only operate on a k-hop subgraph around `node_idx`.
    subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=None, flow=__flow__(model))

    # edge_mask = edge_mask[hard_edge_mask]

    if threshold is not None:
        edge_mask = (edge_mask >= threshold).to(torch.float)

    if y is None:
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    else:
        y = y[subset].to(torch.float) / y.max().item()

    data = Data(edge_index=edge_index, att=edge_mask[edge_mask!=0], y=y,
                num_nodes=y.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)

    node_kwargs = copy(kwargs)
    node_kwargs['node_size'] = kwargs.get('node_size') or 800
    node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

    label_kwargs = copy(kwargs)
    label_kwargs['font_size'] = kwargs.get('font_size') or 10

    pos = nx.spring_layout(G)
    plt.switch_backend("agg")
    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['att']*2, 0.05),
                shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))
    nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
    nx.draw_networkx_labels(G, pos, **label_kwargs)

    return ax, G

def denoise_graph(data, weighted_edge_mask, node_explanations, neighbours, node_idx, feat=None, label=None, threshold_num=10):
    """Cleaning a graph by thresholding its node values.

    Args:
            - weighted_edge_mask:  Edge mask, with importance given to each edge
            - node_explanations :  Shapley values for neighbours
            - neighbours
            - node_idx          :  Index of node to highlight (TODO ?)
            - feat              :  An array of node features.
            - label             :  A list of node labels.
            - theshold_num      :  The maximum number of nodes to threshold.
    """
    # Subgraph with only relevant nodes - pytorch
    edge_index = data.edge_index[:, weighted_edge_mask.nonzero().T[0]]

    s = subgraph(
        torch.cat((torch.tensor([node_idx]), neighbours)), edge_index)[0]
    
    # Disregard size of explanations
    node_explanations = np.abs(node_explanations)

    # Create graph of neighbourhood of node of interest
    G = nx.DiGraph()
    G.add_nodes_from(neighbours.detach().numpy())
    G.add_node(node_idx)
    G.nodes[node_idx]["self"] = 1
    if feat is not None:
        for node in G.nodes():
            G.nodes[node]["feat"] = feat[node].detach().numpy()
    if label is not None:
        for node in G.nodes():
            G.nodes[node]["label"] = label[node].item()

    # Find importance threshold required to retrieve 10 most import nei.
    threshold_num = min(len(neighbours), threshold_num)
    threshold = np.sort(
        node_explanations)[-threshold_num]

    # Add edges 
    # weighted_edge_list = [
    #      (data.edge_index[0, i].item(),
    #       data.edge_index[1, i].item(), weighted_edge_mask[i].item())
    #      for i, _ in enumerate(weighted_edge_mask)
        # if weighted_edge_mask[i] >= threshold
    #  ]
    
    # # Keep edges that satisfy the threshold
    # node_expl_dico = {}
    # for i, imp in enumerate(node_explanations):
    #     node_expl_dico[neighbours[i].item()] = imp 
    # node_expl_dico[node_idx]=torch.tensor(0)
    # weighted_edge_list = [ (el1.item(),el2.item(),node_expl_dico[el1.item()].item()) for el1,el2 in zip(s[0],s[1]) ]
    
    # Add edges 
    imp = weighted_edge_mask[weighted_edge_mask != 0]
    weighted_edge_list = [ (el1.item(), el2.item(),i.item()) for el1, el2, i in (zip(s[0], s[1], imp)) ]
    G.add_weighted_edges_from(weighted_edge_list)

    # Keep nodes that satisfy the threshold
    del_nodes = []
    for i, node in enumerate(G.nodes()):
        if node != node_idx:
            if node_explanations[i] < threshold:
                del_nodes.append(node)
    G.remove_nodes_from(del_nodes)

    return G


def log_graph(G,
              identify_self=True,
              nodecolor="label",
              epoch=0,
              fig_size=(4, 3),
              dpi=300,
              label_node_feat=False,
              edge_vmax=None,
              args=None):
    """
    Args:
            nodecolor: the color of node, can be determined by 'label', or 'feat'. For feat, it needs to
                    be one-hot'
    """
    cmap = plt.get_cmap("Set1")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_colors = []
    # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in G.edges.data('weight', default=1)]
    edge_colors = [w for (u, v, w) in G.edges.data("weight", default=1)]

    # maximum value for node color
    vmax = 8
    for i in G.nodes():
        if nodecolor == "feat" and "feat" in G.nodes[i]:
            num_classes = G.nodes[i]["feat"].size()[0]
            if num_classes >= 10:
                cmap = plt.get_cmap("tab20")
                vmax = 19
            elif num_classes >= 8:
                cmap = plt.get_cmap("tab10")
                vmax = 9
            break

    feat_labels = {}
    for i in G.nodes():
        if identify_self and "self" in G.nodes[i]:
            node_colors.append(0)
        elif nodecolor == "label" and "label" in G.nodes[i]:
            node_colors.append(G.nodes[i]["label"] + 1)
        elif nodecolor == "feat" and "feat" in G.nodes[i]:
            # print(G.nodes[i]['feat'])
            feat = G.nodes[i]["feat"].detach().numpy()
            # idx with pos val in 1D array
            feat_class = 0
            for j in range(len(feat)):
                if feat[j] == 1:
                    feat_class = j
                    break
            node_colors.append(feat_class)
            feat_labels[i] = feat_class
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels = None

    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    #pos_layout = nx.kamada_kawai_layout(G, weight=None)
    pos_layout = nx.fruchterman_reingold_layout(G)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        edge_vmax = 1
        edge_vmin = 0
    else:
        weights = [d for (u, v, d) in G.edges(data="weight", default=1)]
        if edge_vmax is None:
            edge_vmax = statistics.median_high(
                [d for (u, v, d) in G.edges(data="weight", default=1)]
            )
        min_color = min(
            [d for (u, v, d) in G.edges(data="weight", default=1)])
        # color range: gray to black
        edge_vmin = 2 * min_color - edge_vmax

    nx.draw(
        G,
        pos=pos_layout,
        arrows=True,
        with_labels=True,
        font_size=4,
        labels=feat_labels,
        node_color=node_colors,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("Greys"),
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=1.0,
        node_size=120,
        alpha=0.8,
    )
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
            node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
                    node(s).
            num_hops: (int): The number of hops :math:`k`.
            edge_index (LongTensor): The edge indices.
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                    :obj:`edge_index` will be relabeled to hold consecutive indices
                    starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                    :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
            flow (string, optional): The flow direction of :math:`k`-hop
                    aggregation (:obj:`"source_to_target"` or
                    :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
                     :class:`BoolTensor`)
    """

    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def custom_to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                    remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data.__dict__.items():
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G
