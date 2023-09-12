import argparse
from trainer import *
from eval import *

os.chdir('E:/anaconda/envs/pytorch/graduation_project')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='--mode', type=str, help='train or eval')
    parser.add_argument('--input', default="E:/anaconda/envs/pytorch/graduation_project/demo", type=str, help='输入')
    parser.add_argument('--epoch', default=1200, type=int, help='训练的总轮数')
    parser.add_argument('--checkpoints', default=None, type=str, help='保存的参数模型')
    parser.add_argument('--pretrained', default=None, type=str, help='预训练的模型参数')
    args = parser.parse_args()

    if args.mode == "--mode":
        trainer(args.input, args.epoch, args.checkpoints, args.pretrained)
    elif args.mode == "--eval":
        eval_mode(args.input, args.checkpoints)


if __name__ == '__main__':
    main()
