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
    embedder.testing()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if __name__ == '__main__':
    main()
