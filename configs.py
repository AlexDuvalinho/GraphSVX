import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')

    # Utils params
    utils_parser = parser.add_argument_group('utils')
    utils_parser.add_argument('--cuda', dest='cuda', help='CUDA.')
    utils_parser.add_argument('--gpu', default=False, help='whether to use GPU.')
    utils_parser.add_argument("--seed", type=int)
    utils_parser.add_argument("--save", type=str,
                              help="True to save the trained model obtained")

    # Model training params
    training_parser = parser.add_argument_group('training')
    training_parser.add_argument('--max_nodes', dest='max_nodes', type=int,
                     help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    training_parser.add_argument('--method', dest='method',
                     help='Method. Possible values: base, ')
    training_parser.add_argument('--batch_size', dest='batch_size', type=int,
                     help='Batch size.')
    training_parser.add_argument('--epochs', dest='num_epochs', type=int,
                     help='Number of epochs to train.')
    training_parser.add_argument('--train_ratio', dest='train_ratio', type=float,
                     help='Ratio of number of graphs training set to all graphs.')
    training_parser.add_argument('--input_dim', dest='input_dim', type=int,
                     help='Input feature dimension')
    training_parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                     help='Hidden dimension')
    training_parser.add_argument('--output_dim', dest='output_dim', type=int,
                     help='Output dimension')
    training_parser.add_argument('--num_gc_layers', dest='num_gc_layers', type=int,
                     help='Number of graph convolution layers before each pooling')
    training_parser.add_argument('--bn', dest='bn', action='store_const',
                     const=True, default=False,
                     help='Whether batch normalization is used')
    training_parser.add_argument('--dropout', dest='dropout', type=float,
                     help='Dropout rate.')
    training_parser.add_argument('--nobias', dest='bias', action='store_const',
                     const=False, default=True,
                     help='Whether to add bias. Default to True.')
    training_parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                     help='Weight decay regularization constant.')
    training_parser.add_argument('--clip', dest='clip', type=float,
                                 help='Gradient clip value')

    # Evaluation1 params
    eval_noise_parser = parser.add_argument_group('eval noise')
    eval_noise_parser.add_argument("--feat_explainers", type=list, 
                        default=['GraphSVX', 'GNNExplainer', 'GraphLIME',
                                    'LIME', 'SHAP', 'Greedy'],
                        help="Name of the benchmarked explainers among \
                        GraphSVX, SHAP, LIME, GraphLIME, Greedy and GNNExplainer")
    eval_noise_parser.add_argument("--node_explainers", type=list, default=['GraphSVX', 'Greedy', 'GNNExplainer'],
                        help="Name of the benchmarked explainers among Greedy, GNNExplainer and GraphSVX")
    eval_noise_parser.add_argument("--test_samples", type=int,
                        help='number of test samples for evaluation')
    eval_noise_parser.add_argument("--K", type=float,
                        help='proportion of most important features considered, among non zero ones')
    eval_noise_parser.add_argument("--prop_noise_feat", type=float,
                        help='proportion of noisy features')
    eval_noise_parser.add_argument("--prop_noise_nodes", type=float,
                        help='proportion of noisy nodes')
    eval_noise_parser.add_argument("--connectedness", type=str,
                        help='how connected are the noisy nodes we define: low, high or medium')
    eval_noise_parser.add_argument("--evalshap", type=bool,
                        help='True if want to compare GraphSVX with SHAP')
    
    # Explanations params
    parser.add_argument("--model", type=str,
                        help="Name of the GNN: GCN or GAT")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset among Cora, PubMed, syn1-6, Mutagenicity")
    parser.add_argument("--indexes", type=list, default=[0],
                        help="indexes of the nodes/graphs whose prediction are explained")
    parser.add_argument("--hops", type=int,
                        help="number k for k-hops neighbours considered in an explanation")
    parser.add_argument("--num_samples", type=int,
                        help="number of coalitions sampled and used to approx shapley values")
    parser.add_argument("--hv", type=str,
                        help="method used to convert the simplified input to the original input space")
    parser.add_argument("--feat", type=str,
                        help="method used to determine the features considered")
    parser.add_argument("--coal", type=str,
                        help="type of coalition sampler")
    parser.add_argument("--g", type=str,
                        help="surrogate model used to train g on derived dataset")
    parser.add_argument("--multiclass", type=bool,
                        help='False if we consider explanations for the predicted class only')
    parser.add_argument("--regu", type=float,
                        help='None if we do not apply regularisation, \
                        1 if we focus only on features in explanations, 0 for nodes')
    parser.add_argument("--info", type=bool,
                        help='True if want to print info')
    parser.add_argument("--fullempty", type=str,
                        help='True if want to discard full and empty coalitions')
    parser.add_argument("--S", type=int,
                        help='Max size of coalitions sampled in priority and treated specifically')

    # args_hv: compute_pred', 'basic_default', 'neutral', 'graph_classification', 'compute_pred_subgraph'
    # args_feat: 'All', 'Expectation', 'Null'
    # args_coal: 'NewSmarterSeparate', 'SmarterSeparate', 'Smarter', 'Smart', 'Random', 'All'
    # args_g: WLS, 'WLR_sklearn', 'WLR_Lasso'

    parser.set_defaults(dataset='syn1',
                        model='GCN',
                        indexes=[500, 600],
                        num_samples=400,
                        fullempty=None,
                        S=1,
                        hops=3,
                        hv='compute_pred',
                        feat='Null',
                        coal='NewSmarterSeparate',
                        g='WLR_sklearn',
                        multiclass=False,
                        regu=None,
                        info=False,
                        seed=10,
                        gpu=False,
                        cuda='0',
                        save=False,
                        feat_explainers=['GraphSVX', 'GNNExplainer', 
                                            'GraphLIME', 'LIME', 'SHAP'],
                        node_explainers=['GraphSVX', 'GNNExplainer', 'Greedy'],
                        test_samples=50,
                        K=0.20,
                        prop_noise_feat=0.20,
                        prop_noise_nodes=0.20,
                        connectedness='medium',
                        opt='adam',   # opt_parser
                        max_nodes=100,
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_gc_layers=3,
                        dropout=0.0,
                        weight_decay=0.005,
                        method='base'
                        )

    return parser.parse_args()

