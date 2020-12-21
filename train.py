import argparse
import os
import torch
import json
from train_helper import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--load-args',
                        help='file to read program args from.')
    args = parser.parse_args()
    with open(args.load_args) as f:
        args = json.load(f)
    with open('args/dataset_paths.json') as f:
        datargs = json.load(f)
    datargs = datargs[args['dataset']]
    return args, datargs


if __name__ == '__main__':
    args, datargs = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args['device'].strip()  # set vis gpu
    trainer = Trainer(args, datargs)
    trainer.setup()
    trainer.train()
