import scipy.sparse as sp
import numpy as np
import torch
import data_load
import utils
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

class embedder:
    def __init__(self, args):
        if args.gpu == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        # args.device = 'cpu'
        # Load data - Cora, CiteSeer, cora_full
        self._dataset = data_load.Dataset(root="data", dataset=args.dataset, is_normalize=args.is_normalize, add_self_loop=args.add_sl)

        self.edge_index = self._dataset.edge_index
        adj = self._dataset.adj

        self.degree = torch.zeros(torch.max(self.edge_index)+1,dtype=torch.float)
        for i in self.edge_index.T: 
            self.degree[i[0]] += 1
            self.degree[i[1]] += 1

        tail_adj = self._dataset.tail_adj
        features = self._dataset.features
        labels = self._dataset.labels

        im_class_num = args.im_class_num
        min_class_num = 10000000
        idx_train_min = None
        idx_train_max = None

        # Natural Setting
        if args.im_ratio == 1:
            args.criterion = 'mean'
            labels, og_to_new = utils.refine_label_order(labels)
            idx_train, idx_val, idx_test, class_num_mat = utils.split_natural(labels, og_to_new)
            samples_per_label = torch.tensor(class_num_mat[:,0])
            # import pdb;pdb.set_trace()

        # Manual Setting
        
        elif args.im_ratio in [0.2, 0.1, 0.05]:
            args.criterion = 'max'
            labels, og_to_new = utils.refine_label_order(labels)
            c_train_num = []
            for i in range(labels.max().item() + 1):
                class_sample_num = (labels==i).nonzero().shape[0]
                if class_sample_num < min_class_num:
                    min_class_num = class_sample_num
                if i > labels.max().item() - im_class_num:  # last classes belong to minority classes
                    c_train_num.append(int(class_sample_num*args.im_ratio*0.8))
#                     c_train_num.append(int(class_sample_num*0.8))
                else:
                    c_train_num.append(int(class_sample_num*0.8))
            idx_train, part_idx_train, idx_train_min, idx_train_max, idx_val, idx_test, class_num_mat = utils.split_manual(labels, c_train_num, og_to_new, min_class_num)
            samples_per_label = torch.tensor(class_num_mat[:,0])

        if 'lte4g' in args.embedder:
            manual = True if args.im_ratio in [0.2, 0.1, 0.05] else False
            idx_train_set_class, ht_dict_class = utils.separate_ht(samples_per_label, labels, idx_train, method=args.sep_class, manual=manual)
            idx_train_set, degree_dict, degrees, above_head, below_tail  = utils.separate_class_degree(adj, idx_train_set_class, below=args.sep_degree)
            
            idx_val_set = utils.separate_eval(idx_val, labels, ht_dict_class, degrees, above_head, below_tail)
            idx_test_set = utils.separate_eval(idx_test, labels, ht_dict_class, degrees, above_head, below_tail)

            args.sep_point = len(ht_dict_class['H'])

            self.idx_train_set_class = idx_train_set_class
            self.degrees = degrees
            self.above_head = above_head
            self.below_tail = below_tail
            # import pdb;pdb.set_trace()

            print('Above Head Degree:', above_head)
            print('Below Tail Degree:', below_tail)
            
            self.idx_train_set = {}
            self.idx_val_set = {}
            self.idx_test_set = {}
            for sep in ['HH', 'HT', 'TH', 'TT']:
                self.idx_train_set[sep] = idx_train_set[sep].to(args.device)
                self.idx_val_set[sep] = idx_val_set[sep].to(args.device)
                self.idx_test_set[sep] = idx_test_set[sep].to(args.device)

        adj = utils.normalize_adj(adj) if args.adj_norm_1 else utils.normalize_sym(adj)
        self.adj = adj.to(args.device)

        tail_adj = utils.normalize_adj(tail_adj) if args.adj_norm_1 else utils.normalize_sym(tail_adj)
        self.tail_adj = tail_adj.to(args.device)

        self.features = features.to(args.device)
        self.labels = labels.to(args.device)
        # self.class_sample_num = class_sample_num
        self.im_class_num = im_class_num

        self.idx_train = idx_train.to(args.device)
        self.part_idx_train = part_idx_train.to(args.device)
        if idx_train_min is not None:
            self.idx_train_min = idx_train_min.to(args.device)
            self.idx_train_max = idx_train_max.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.samples_per_label = samples_per_label
        self.class_num_mat = class_num_mat
        print(class_num_mat)

        args.nfeat = features.shape[1]
        args.nclass = labels.max().item() + 1
        args.im_class_num = im_class_num

        self.args = args
