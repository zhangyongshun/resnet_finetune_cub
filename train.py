import os
import argparse
import torch
from utils.Config import Config
from trainer import NetworkManager


def main():
    parser = argparse.ArgumentParser(
        description='Options for base model finetuning on CUB_200_2011 datasets'
    )
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='base learning rate for training')
    parser.add_argument('--net_choice', type=str, required=True,
                        help='net_choice for choosing network, whose value is in ["ResNet"]')
    parser.add_argument('--model_choice', type=int, required=True,
                        help='model_choice for choosing depth of network, whose value is in [50, 101, 152]')
    parser.add_argument('--epochs', type=int, default=95,
                        help='batch size for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay for SGD')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='choose one gpu for training')
    parser.add_argument('--img_size', type=int, default=448,
                        help='image\'s size for transforms')
    args = parser.parse_args()
    assert args.gpu_id.__class__ == int


    options = {
        'net_choice': args.net_choice,
        'model_choice': args.model_choice,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'img_size': args.img_size,
        'device': torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    }

    path = {
        'data': Config.data_path,
        'model_save': Config.model_save_path
    }

    for p in path:
        print(p)
        print(path[p])
        assert os.path.isdir(path[p])

    manager = NetworkManager(options, path)
    manager.train()


if __name__ == '__main__':
    main()
