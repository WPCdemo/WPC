import time
import arg
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

def main():
    parser = arg.get_parser()
    args = parser.parse_args()

    if args.embedder == 'lte4g':
        from models.lte4g import lte4g
        print(args.layer)
        embedder = lte4g(args)
        

    else:
        if args.embedder == 'origin':
            from models.baseline import origin
            embedder = origin(args)

        elif args.embedder == 'oversampling':
            from models.baseline import oversampling
            embedder = oversampling(args)
        
        elif args.embedder == 'reweight':
            from models.baseline import reweight
            embedder = reweight(args)

        elif args.embedder == 'smote':        
            from models.baseline import smote
            embedder = smote(args)

        elif args.embedder == 'embed_smote':
            from models.baseline import embed_smote
            embedder = embed_smote(args)

        elif args.embedder == 'graphsmote_T':
            from models.baseline import graphsmote_T
            embedder = graphsmote_T(args)

        elif args.embedder == 'graphsmote_O':
            from models.baseline import graphsmote_O
            embedder = graphsmote_O(args)
        elif args.embedder == 'intent':
            from models.baseline import Intent
            embedder = Intent(args)
        elif args.embedder == 'intent_kmeans':
            from models.baseline import Intent_kmeans
            embedder = Intent_kmeans(args)
        elif args.embedder == 'geometer':
            from models.baseline import Geometer
            embedder = Geometer(args)
        
    t_total = time.time()
    embedder.training()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if __name__ == '__main__':
    main()



# def cmp(x, y):
#     # matrix = torch.mm(a, a)
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)


#     x1 = x.unsqueeze(1).expand(n, m, d)
#     y1 = y.unsqueeze(0).expand(n, m, d)
#     # return cos_similar(x1, y1)
#     # return 1/(1+torch.exp(-100 *(x1-y1)))
#     return torch.pow(x1 - y1, 2).sum(2)  # N x M
# def cmp1(x, y):
#     # matrix = torch.mm(a, a)
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     assert d == y.size(1)


#     x1 = x.unsqueeze(1).expand(n, m, d)
#     y1 = y.unsqueeze(0).expand(n, m, d)
#     # return cos_similar(x1, y1)
#     # return 1/(1+torch.exp(-100 *(x1-y1)))
#     return y1 - x1
# import torch
# a = torch.randn(10, 100)
# center = torch.randn(1, 100)
# b = cmp(a, center)
# b1 = 1/(1+torch.exp(-10000 *cmp1(b, b)))
# b1 = b1.squeeze(-1)
# rank = torch.sum(b1, 1)-0.5
# import pdb;pdb.set_trace()