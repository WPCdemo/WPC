import time
import arg
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main():
    parser = arg.get_parser()
    args = parser.parse_args()

    if args.embedder == 'wpc':
        from models import WPC
        embedder = WPC(args)

    t_total = time.time()
    embedder.training()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


if __name__ == '__main__':
    main()
