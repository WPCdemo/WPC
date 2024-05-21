import argparse


def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')


def get_parser():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--baseline', action='store_true', default=False, help="Run")
    parser.add_argument('--gpu', type=int, default=0, help="Choose GPU number")
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'pubmed', 'wikics', 'Cora_full'])
    parser.add_argument('--im_class_num', type=int, default=1, help="Number of tail classes")
    parser.add_argument('--im_ratio', type=float, default=0.1, help="1 for natural, [0.2, 0.1, 0.05] for manual, 0.01 for LT setting")
    parser.add_argument('--layer', type=str, default='gat', choices=['gcn', 'gat'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rw', type=float, default=0.000001, help="Balances edge loss within node classification loss")
    parser.add_argument('--ep_pre', type=int, default=0, help="Number of epochs to pretrain.")
    parser.add_argument('--ep', type=int, default=2000, help="Number of epochs to train.")
    parser.add_argument('--ep_early', type=int, default=1000, help="Early stop criterion.")
    parser.add_argument('--add_sl', type=str2bool, default=True, help="Whether to include self-loop")
    parser.add_argument('--adj_norm_1', action='store_true', default=True, help="D^(-1)A")
    parser.add_argument('--adj_norm_2', action='store_true', default=False, help="D^(-1/2)AD^(-1/2)")
    parser.add_argument('--nhid', type=int, default=64, help="Number of hidden dimensions")
    parser.add_argument('--nhead', type=int, default=1, help="Number of multi-heads")
    parser.add_argument('--wd', type=float, default=5e-4, help="Controls weight decay")
    parser.add_argument('--num_seed', type=int, default=10, help="Number of total seeds") 
    parser.add_argument('--is_normalize', action='store_true', default=False, help="Normalize features")
    parser.add_argument('--cls_og', type=str, default='GNN', choices=['GNN', 'MLP'], help="Wheter to user (GNN+MLP) or (MLP) as a classifier")

    parser.add_argument('--n_factors',  default=5)
    parser.add_argument('--n_hops',  default=2)
    parser.add_argument('--ind',  default='distance')
    parser.add_argument('--num_intent_cluster',  default=10)
    parser.add_argument("--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied.")


    parser.add_argument('--embedder', nargs='?', default='origin', choices=['origin', 'reweight', 'oversampling', 'smote', 'embed_smote', 'graphsmote_T', 'graphsmote_O', 'intent', 'intent_kmeans', 'wpc'])
    parser.add_argument('--up_scale', type=float, default=1, help="Scale of Oversampling")
    parser.add_argument('--up_scale1', type=float, default=5, help="Scale of Oversampling")

    return parser
