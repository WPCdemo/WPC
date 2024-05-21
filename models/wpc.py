from embedder import embedder
import torch.nn as nn
import layers
import torch.optim as optim
import utils
import torch.nn.functional as F
from scipy.spatial import ConvexHull
import torch
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import os
from sklearn.metrics import f1_score, classification_report, precision_score
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_proto_embedding(adj, proto_model, spt_idx_list, embeddings, degree_list=None):
    for i, spt_idx in enumerate(spt_idx_list):
        spt_embedding_i = embeddings[spt_idx]
       
        degree_list = degree_list.to(adj.device)
        if degree_list == None:
            proto_embedding_i, _ = proto_model(spt_embedding_i)#torch.Size([1, 512])
        else:
            degree_list_i = degree_list[spt_idx]
            proto_embedding_i, _ = proto_model(spt_embedding_i, degree_list_i)
        if i == 0:
            proto_embedding = proto_embedding_i
        else:
            proto_embedding = torch.cat((proto_embedding, proto_embedding_i), dim=0)
        
    return proto_embedding


class ProtoRepre(nn.Module):
    def __init__(self, args):
        super(ProtoRepre, self).__init__()

    
    def forward(self, spt_embedding_i, degree_list_i=None):
        degree_list_i = degree_list_i.to(spt_embedding_i.device)
        if degree_list_i == None:
            avg_proto_i = torch.sum(spt_embedding_i, 0) / spt_embedding_i.shape[0]

        else:
            norm_degree = degree_list_i / torch.sum(degree_list_i)
            norm_degree = norm_degree.unsqueeze(1)
            avg_proto_i = torch.sum(
            torch.mul(spt_embedding_i, norm_degree), 0
            )

        return avg_proto_i.unsqueeze(0), None



class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(targets.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
      

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def ProximityLoss(mask, qry_embedding, proto_embedding, labels, im_class_num, up_scale, focal=False, class_num=6, base_class_num=3):
    dists = euclidean_dist(qry_embedding, proto_embedding)
    if focal:
        alpha = torch.tensor([1.0] * base_class_num + [0.1] * (class_num - base_class_num))
        focal_loss = FocalLoss(class_num, alpha=alpha, gamma=1)

        loss = focal_loss(-dists, labels)

    else:

        weight = mask
        weight[weight==1] = 1.5
        weight[weight==0] = 1
        weight = weight.detach()

        log_probs = torch.log_softmax(-dists, dim=1)
        loss = torch.sum(-log_probs[torch.arange(labels.size(0)), labels]*weight)/weight.shape[0]

    return loss

def UniformityLoss(proto_embedding, device):
    center_proto_embedding = torch.mean(proto_embedding, dim=0).unsqueeze(0)
    normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding)

    cos_dist_matrix = torch.mm(normalize_proto_embedding, normalize_proto_embedding.T)
    unit_matrix = torch.eye(cos_dist_matrix.shape[0]).to(device)

    cos_dist_matrix = cos_dist_matrix - unit_matrix

    loss = torch.max(cos_dist_matrix, 1).values
    return torch.mean(loss)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    acc = precision_score(labels, preds, average='macro')

    return acc


def get_idx(labels, idx_train):
    proto_idx = []
    max_class = max(labels)
    for i in range(max_class+1):
        proto_idx.append(idx_train[labels==i])
    return proto_idx

def plot_embedding(data, label,  title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def get_adj(args, labels, idx_train):
    all_adj = []
    for i in range(max(labels)+1):
        class_num = len(idx_train[labels==i])
        random_adj = torch.rand(class_num, class_num) - torch.eye(class_num, class_num)
        random_adj[random_adj>0.5] = 1
        random_adj[random_adj<=0.5] = 0

        adj = random_adj
        
        indices = torch.arange(class_num)
        indices = indices.repeat(class_num,1)
        indices = torch.flatten(indices.T).numpy()

        flatten_adj = torch.flatten(adj).numpy()
       
        adj = sp.coo_matrix((np.ones(class_num*class_num), (indices, flatten_adj)), shape=(class_num, class_num), dtype=np.float32)
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
        adj = utils.normalize_adj(adj) if args.adj_norm_1 else utils.normalize_sym(adj)
        adj = adj.to(args.device)
        all_adj.append(adj)

    return all_adj




def SeparabilityLoss(proto_embedding):

    dist_matrix = euclidean_dist(proto_embedding, proto_embedding)


    min_dist = torch.exp(-1 * torch.min(dist_matrix, dim=1).values)
    loss = torch.mean(min_dist)
    return loss


def generate(embeddings, labels, proto_embedding):
    total_embed = None
    total_label = None
    top_k = 10000
    for i in range(torch.max(labels)+1):

        embeddings_part = embeddings[labels==i]
        if top_k > len(embeddings_part):
            top_k = len(embeddings_part)
    for i in range(torch.max(labels)+1):

        embeddings_part = embeddings[labels==i]
        labels_part = labels[labels==i]
        proto_embedding_part = proto_embedding[i].unsqueeze(0)
        n = embeddings_part.size(0)
        m = proto_embedding_part.size(0)
        d = embeddings_part.size(1)
        assert d == proto_embedding_part.size(1)


        embeddings_part1 = embeddings_part.unsqueeze(1).expand(n, m, d)
        proto_embedding_part1 = proto_embedding_part.unsqueeze(0).expand(n, m, d)
        distance = torch.pow(embeddings_part1 - proto_embedding_part1, 2).sum(2)  # N x M

        n = distance.size(0)
        m = distance.size(0)
        d = distance.size(1)
        assert d == distance.size(1)


        distance1 = distance.unsqueeze(1).expand(n, m, d)
        distance2 = distance.unsqueeze(0).expand(n, m, d)
        distance = distance2 - distance1
        rank = 1/(1+torch.exp(-100000 *distance))
        rank = rank.squeeze(-1)
        rank = torch.sum(rank, 1) - 0.5
        embeddings_part = embeddings_part[rank<top_k]
        labels_part1 = torch.cat((labels_part[0].unsqueeze(0), labels_part[:len(embeddings_part)]), 0)

        if total_embed is None:
            total_embed = torch.cat((embeddings_part, proto_embedding_part), 0)
            total_label = labels_part1
        else:
            embeddings_part = torch.cat((embeddings_part, proto_embedding_part), 0)
            total_embed = torch.cat((total_embed, embeddings_part), 0)
            total_label = torch.cat((total_label, labels_part1), 0)
    
    shuffle_id = np.arange(len(total_embed)-1)
    np.random.shuffle(shuffle_id)
    shuffle_id = torch.tensor(shuffle_id)

    total_embed = total_embed[shuffle_id]
    total_label = total_label[shuffle_id]

    return total_embed, total_label


def generate1(args, embeddings, labels, proto_embedding):
    
    n = embeddings.size(0)
    m = proto_embedding.size(0)
    d = embeddings.size(1)
    assert d == proto_embedding.size(1)


    embeddings1 = embeddings.unsqueeze(1).expand(n, m, d)
    proto_embedding1 = proto_embedding.unsqueeze(0).expand(n, m, d)
    distance = torch.pow(embeddings1 - proto_embedding1, 2).sum(2)  # N x M
    distance_labels = distance.argmin(dim=1)

    weight = torch.ones_like(labels).long()

    weight[distance_labels!=labels] += args.up_scale
    return weight

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    assert d == y.size(1)


    x1 = x.unsqueeze(1).expand(n, m, d)
    y1 = y.unsqueeze(0).expand(n, m, d)
    

    return torch.pow(x1 - y1, 2).sum(2)  # N x M

def compute_mask(length, embed, p, idx, C, delta=1000):
    m = torch.zeros(length, dtype=torch.float32).to(idx[0].device)
    for k in range(C):
        indices = [i for i in range(C) if i != k]

        dist1 = euclidean_dist(embed[k], p[k].unsqueeze(0))
        
        dist2 = euclidean_dist(embed[k], p[indices, :]).min(dim=1)[0].unsqueeze(-1)

        sigma = 1 / (1 + torch.exp(-delta * (dist2 - dist1)))

        m[idx[k]] = sigma.sum(dim=1)

    return m


def compute_ranking(length, embed, p, idx, C, delta=1000):
    mask = torch.zeros(length, dtype=torch.float32).to(idx[0].device)
    rank = torch.zeros((C, length), dtype=torch.float32).to(idx[0].device)
    for k in range(C):
        indices = [i for i in range(C) if i != k]
        dist = euclidean_dist(embed[k], p[k].unsqueeze(0))

        dim0 = dist.shape[0]
        dim1 = dist.shape[1]
        x = dist.unsqueeze(0).expand(dim0, dim0, dim1)
        diff = x - x.transpose(0, 1)

        sigma = 1 / (1 + torch.exp(-delta * diff))
        sigma[sigma==0.5]=0.0
        rank[k, idx[k]] = sigma.sum(dim=1).sum(-1)

        
        mask[idx[k][(rank[k, idx[k]]>0) & (rank[k, idx[k]]<=160)]] = 1

    return mask


class WPC():
    def __init__(self, args):
        self.args = args

    def training(self):
        self.args.embedder = f'({self.args.layer.upper()})' + self.args.embedder + f'_cls_{self.args.cls_og}'
        if self.args.im_ratio == 1: # natural
            os.makedirs(f'./results/baseline/natural/{self.args.dataset}', exist_ok=True)
            text = open(f'./results/baseline/natural/{self.args.dataset}/{self.args.embedder}.txt', 'w')
        else: # manual
            os.makedirs(f'./results/baseline/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}', exist_ok=True)
            text = open(f'./results/baseline/manual/{self.args.dataset}/{self.args.im_class_num}/{self.args.im_ratio}/{self.args.embedder}.txt', 'w')

        seed_result = {}
        seed_result['acc'] = []
        seed_result['macro_F'] = []
        seed_result['gmeans'] = []
        seed_result['bacc'] = []
        seed_result['acc_tail'] = []
        seed_result['macro_F_tail'] = []
        seed_result['gmeans_tail'] = []
        seed_result['bacc_tail'] = []
        seed_result['acc_head'] = []
        seed_result['macro_F_head'] = []
        seed_result['gmeans_head'] = []
        seed_result['bacc_head'] = []
        
        for seed in range(5, 5+self.args.num_seed):
            print(f'============== seed:{seed} ==============')
            utils.seed_everything(seed)
            print('seed:', seed, file=text)
            self = embedder(self.args)

            model = modeler(self.args, self.adj).to(self.args.device)
            optimizer_fe = optim.Adam(model.encoder.parameters(), lr=self.args.lr, weight_decay=self.args.wd)  # feature extractor
            optimizer_cls = optim.Adam(model.classifier.parameters(), lr=self.args.lr, weight_decay=self.args.wd)  # node classifier

            proto_model = ProtoRepre(self.args).to(self.args.device)

            # Main training
            val_f = []
            test_results = []

            best_metric = 0

            for epoch in range(self.args.ep):
                model.train()

                optimizer_fe.zero_grad()
                optimizer_cls.zero_grad()

                embeddings, loss_cls = model(self.features, self.labels, self.idx_train, self.idx_train_min, self.idx_train_max)
                labels = self.labels[self.idx_train]
                self.part_idx_train = self.idx_train
                labels_part = self.labels[self.part_idx_train]

                proto_train_idx = get_idx(labels, self.idx_train)
                part_proto_train_idx = get_idx(labels_part, self.part_idx_train)
                proto_embedding = get_proto_embedding(self.adj, proto_model, part_proto_train_idx, embeddings, self.degree)
                



                embeddings_part = embeddings[self.part_idx_train]
                


                
                weight = generate1(self.args, embeddings_part, labels_part, proto_embedding)
               

                embed = []
                idxs = []
                for i in range(len(part_proto_train_idx)):
                    embed.append(embeddings[part_proto_train_idx[i]])
                    mask = torch.isin(self.part_idx_train, part_proto_train_idx[i])
                    idx = torch.nonzero(mask).squeeze()
                    idxs.append(idx)

                    
                    
                mask = compute_mask(embeddings_part.shape[0], embed, proto_embedding, idxs, len(part_proto_train_idx), delta=1000000)
                rank = compute_ranking(embeddings_part.shape[0], embed, proto_embedding, idxs, len(part_proto_train_idx), delta=1000000)
                mask = mask + rank
                mask[mask==2] = 1
                

                p_loss = ProximityLoss(mask, embeddings_part, proto_embedding, labels_part, self.args.im_class_num, self.args.up_scale)
                u_loss = UniformityLoss(proto_embedding, self.args.device)

                loss = 1 * p_loss + 0.1 * u_loss + loss_cls

                loss.backward()

                optimizer_fe.step()
                optimizer_cls.step()


                # Evaluation
                model.eval()
                embed = model.encoder(self.features)
                output = model.classifier(embed)
                embeddings_val = embed[self.idx_val]

                dists = euclidean_dist(embeddings_val, proto_embedding)

                acc_val, macro_F_val, gmeans_val, bacc_val = utils.performance_measure(output[self.idx_val], self.labels[self.idx_val], pre='valid')

                val_f.append(macro_F_val)
                max_idx = val_f.index(max(val_f))

                if best_metric <= macro_F_val:
                    best_metric = macro_F_val
                    best_model = deepcopy(model)
                    best_prototype = proto_embedding

                # Test
                embeddings_test = embed[self.idx_test]

                dists = euclidean_dist(embeddings_test, proto_embedding)
                acc_test, macro_F_test, gmeans_test, bacc_test= utils.performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')

                test_results.append([acc_test, macro_F_test, gmeans_test, bacc_test])
                best_test_result = test_results[max_idx]

                st = "[seed {}][{}][Epoch {}]".format(seed, self.args.embedder, epoch)
                st += "[Val] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}|| ".format(acc_val, macro_F_val, gmeans_val, bacc_val)
                st += "[Test] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}\n".format(acc_test, macro_F_test, gmeans_test, bacc_test)
                st += "  [*Best Test Result*][Epoch {}] ACC: {:.1f}, Macro-F1: {:.1f}, G-Means: {:.1f}, bACC: {:.1f}".format(max_idx, best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3])
                    
                if epoch % 100 == 0:
                    print(st)

                if (epoch - max_idx > self.args.ep_early) or (epoch+1 == self.args.ep):
                    if epoch - max_idx > self.args.ep_early:
                        print("Early stop")
                    embed = best_model.encoder(self.features)
                    output = best_model.classifier(embed)
                    embeddings_test = embed[self.idx_test]
                    
                    self.idx_test_tail = self.idx_test[self.labels[self.idx_test]>=3]
                    self.idx_test_head = self.idx_test[self.labels[self.idx_test]<3]
                    dists = euclidean_dist(embeddings_test, best_prototype)
                    best_test_result[0], best_test_result[1], best_test_result[2], best_test_result[3] = utils.performance_measure(output[self.idx_test], self.labels[self.idx_test], pre='test')
                    best_test_result_tail0, best_test_result_tail1, best_test_result_tail2, best_test_result_tail3 = utils.performance_measure(output[self.idx_test_tail], self.labels[self.idx_test_tail], pre='test')
                    best_test_result_head0, best_test_result_head1, best_test_result_head2, best_test_result_head3 = utils.performance_measure(output[self.idx_test_head], self.labels[self.idx_test_head], pre='test')
                    break

            seed_result['acc'].append(float(best_test_result[0]))
            seed_result['macro_F'].append(float(best_test_result[1]))
            seed_result['gmeans'].append(float(best_test_result[2]))
            seed_result['bacc'].append(float(best_test_result[3]))
            
            seed_result['acc_head'].append(float(best_test_result_head0))
            seed_result['macro_F_head'].append(float(best_test_result_head1))
            seed_result['gmeans_head'].append(float(best_test_result_head2))
            seed_result['bacc_head'].append(float(best_test_result_head3))
            
            seed_result['acc_tail'].append(float(best_test_result_tail0))
            seed_result['macro_F_tail'].append(float(best_test_result_tail1))
            seed_result['gmeans_tail'].append(float(best_test_result_tail2))
            seed_result['bacc_tail'].append(float(best_test_result_tail3))
        acc = seed_result['acc']
        f1 = seed_result['macro_F']
        gm = seed_result['gmeans']
        bacc = seed_result['bacc']

        print('[Averaged result] ACC: {:.1f}+{:.1f}, Macro-F: {:.1f}+{:.1f}, G-Means: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc), np.std(bacc)))
        print('\n[Averaged head result] ACC: {:.1f}+{:.1f}, Macro-F: {:.1f}+{:.1f}, G-Means: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}'.format(np.mean(seed_result['acc_head']), np.std(seed_result['acc_head']), np.mean(seed_result['macro_F_head']), np.std(seed_result['macro_F_head']), np.mean(seed_result['gmeans_head']), np.std(seed_result['gmeans_head']), np.mean(seed_result['bacc_head']), np.std(seed_result['bacc_head'])))
        print('\n[Averaged tail result] ACC: {:.1f}+{:.1f}, Macro-F: {:.1f}+{:.1f}, G-Means: {:.1f}+{:.1f}, bACC: {:.1f}+{:.1f}'.format(np.mean(seed_result['acc_tail']), np.std(seed_result['acc_tail']), np.mean(seed_result['macro_F_tail']), np.std(seed_result['macro_F_tail']), np.mean(seed_result['gmeans_tail']), np.std(seed_result['gmeans_tail']), np.mean(seed_result['bacc_tail']), np.std(seed_result['bacc_tail'])))
        print(file=text)
        print('ACC Macro-F G-Means bACC', file=text)
        print('{:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f} {:.1f}+{:.1f}'.format(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(gm), np.std(gm), np.mean(bacc), np.std(bacc)), file=text)
        print(file=text)
        print(self.args, file=text)
        print(self.args)
        text.close()

class modeler(nn.Module):
    def __init__(self, args, adj):
        super(modeler, self).__init__()
        self.args = args

        self.encoder = layers.GNN_Encoder(layer=args.layer, nfeat=args.nfeat, nhid=args.nhid, nhead=args.nhead, dropout=args.dropout, adj=adj)
        if args.cls_og == 'GNN':
            self.classifier = layers.GNN_Classifier(layer=args.layer, nhid=args.nhid, nclass=args.nclass, nhead=args.nhead, dropout=args.dropout, adj=adj)
        elif args.cls_og == 'MLP':
            self.classifier = layers.MLP(nhid=args.nhid, nclass=args.nclass)

    def forward(self, features, labels, idx_train, idx_train_min, idx_train_max):
        embed = self.encoder(features)
        output = self.classifier(embed)
        weight = features.new((labels.max().item() + 1)).fill_(1)
        weight[-self.args.im_class_num:] = 1 + self.args.up_scale

        loss_nodeclassification = F.cross_entropy(output[idx_train_max], labels[idx_train_max])

        return embed, loss_nodeclassification
