""" models.py

	Define 2 GNN models made of GCN and GAT layers respectively
"""

import numpy as np
from torch.nn import init
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_adj
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    """
    Construct a GNN with several Graph Convolution blocks
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.conv_in = GCNConv(input_dim, hidden_dim[0])
        self.conv = [GCNConv(hidden_dim[i-1], hidden_dim[i])
                     for i in range(1, len(hidden_dim))]
        self.conv_out = GCNConv(hidden_dim[-1], output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv_in(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for block in self.conv:
            x = F.relu(block(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv_out(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    """
    Contruct a GNN with several Graph Attention layers 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.conv_in = GATConv(
            input_dim, hidden_dim[0], heads=n_heads[0], dropout=self.dropout)
        self.conv = [GATConv(hidden_dim[i-1] * n_heads[i-1], hidden_dim[i],
                             heads=n_heads[i], dropout=self.dropout) for i in range(1, len(n_heads)-1)]
        self.conv_out = GATConv(
            hidden_dim[-1] * n_heads[-2], output_dim, heads=n_heads[-1], dropout=self.dropout, concat=False)

    def forward(self, x, edge_index, att=None):
        x = F.dropout(x, p=self.dropout, training=self.training)

        if att:  # if we want to see attention weights
            x, alpha = self.conv_in(
                x, edge_index, return_attention_weights=att)
            x = F.elu(x)

            for attention in self.conv:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(attention(x, edge_index))

            x = F.dropout(x, p=self.dropout, training=self.training)
            x, alpha2 = self.conv_out(
                x, edge_index, return_attention_weights=att)

            return F.log_softmax(x, dim=1), alpha, alpha2

        else:  # we don't consider attention weights
            x = self.conv_in(x, edge_index)
            x = F.elu(x)

            for attention in self.conv:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(attention(x, edge_index))

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv_out(x, edge_index)

            return F.log_softmax(x, dim=1)


class LinearRegressionModel(nn.Module):
    """Construct a simple linear regression

    """    
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred =self.linear1(x)
        return y_pred



class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, add_self=False, args=None):
        super(GCNNet, self).__init__()
        self.input_dim = input_dim
        print('GCNNet input_dim:', self.input_dim)
        self.hidden_dim = hidden_dim
        print('GCNNet hidden_dim:', self.hidden_dim)
        self.label_dim = label_dim
        print('GCNNet label_dim:', self.label_dim)
        self.num_layers = num_layers
        print('GCNNet num_layers:', self.num_layers)

        self.args = args
        self.dropout = dropout
        self.act = F.relu
        #self.celloss = torch.nn.CrossEntropyLoss()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.input_dim, self.hidden_dim))
        for layer in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        print('len(self.convs):', len(self.convs))

        self.linear = torch.nn.Linear(
            len(self.convs) * self.hidden_dim, self.label_dim)
        
        # Init weights
        # torch.nn.init.xavier_uniform_(self.linear.weight.data)
        #for conv in self.convs:
        #    torch.nn.init.xavier_uniform_(conv.weight.data)  

    def forward(self, x, edge_index):
        x_all = []

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_all.append(x)
        x = torch.cat(x_all, dim=1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)



# GCN basic operation
class GraphConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        add_self=False,
        normalize_embedding=False,
        dropout=0.0,
        bias=True,
        gpu=True,
        att=False,
    ):
        super(GraphConv, self).__init__()
        self.att = att
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not gpu:
            self.weight = nn.Parameter(
                torch.FloatTensor(input_dim, output_dim))
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim)
                )
            if att:
                self.att_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim))
        else:
            self.weight = nn.Parameter(
                torch.FloatTensor(input_dim, output_dim).cuda())
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim).cuda()
                )
            if att:
                self.att_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim).cuda()
                )
        if bias:
            if not gpu:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        if self.att:
            x_att = torch.matmul(x, self.att_weight)
            att = x_att @ x_att.permute(0, 2, 1)
            adj = adj * att

        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.add_self:
            self_emb = torch.matmul(x, self.self_weight)
            y += self_emb
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)

        return y, adj


class GcnEncoderGraph(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers,
        pred_hidden_dims=[],
        concat=True,
        bn=True,
        dropout=0.0,
        add_self=False,
        args=None,
    ):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = add_self
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.celloss = nn.CrossEntropyLoss()

        self.bias = True
        self.gpu = args.gpu
        if args.method == "att":
            self.att = True
        else:
            self.att = False
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=True,
            dropout=dropout,
        )
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs
        )

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.att:
                    init.xavier_uniform_(
                        m.att_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.add_self:
                    init.xavier_uniform_(
                        m.self_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        num_layers,
        add_self,
        normalize=False,
        dropout=0.0,
    ):
        conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        conv_block = nn.ModuleList(
            [
                GraphConv(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    add_self=add_self,
                    normalize_embedding=normalize,
                    dropout=dropout,
                    bias=self.bias,
                    gpu=self.gpu,
                    att=self.att,
                )
                for i in range(num_layers - 2)
            ]
        )
        conv_last = GraphConv(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        return conv_first, conv_block, conv_last

    def build_pred_layers(
        self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1
    ):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        """ For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        """
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def gcn_forward(
        self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None
    ):
        """ Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
            The embedding dim is self.pred_input_dim
        """

        x, adj_att = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        adj_att_all = [adj_att]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x, _ = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(adj_att)
        x, adj_att = conv_last(x, adj)
        x_all.append(x)
        adj_att_all.append(adj_att)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        return x_tensor, adj_att_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(
                max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x, adj_att = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        adj_att_all = [adj_att]
        for i in range(self.num_layers - 2):
            x, adj_att = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
            adj_att_all.append(adj_att)
        x, adj_att = self.conv_last(x, adj)
        adj_att_all.append(adj_att)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)

        self.embedding_tensor = output
        ypred = self.pred_model(output)
        
        return F.log_softmax(ypred, dim=1)


class GcnEncoderNode(GcnEncoderGraph):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        num_layers,
        pred_hidden_dims=[],
        concat=True,
        bn=True,
        dropout=0.0,
        args=None,
    ):
        super(GcnEncoderNode, self).__init__(
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims,
            concat,
            bn,
            dropout,
            args=args,
        )

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        # Convert ajd matrix
        adj = to_dense_adj(adj, max_num_nodes=x.shape[0])

        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(
                max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.adj_atts = []
        self.embedding_tensor, adj_att = self.gcn_forward(
            x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
        )
        pred = self.pred_model(self.embedding_tensor)
        pred = pred.squeeze(0)

        return F.log_softmax(pred, dim=1)  # pred


    def loss(self, output, target):
        return self.celloss(output, target)
