import threading
import configparser
import random
import wandb

from clients import StereoClient
from server import StereoServer

import argparse
import torch
import numpy as np

import wandb

parser = argparse.ArgumentParser(description='FedStereo')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--nodelist', type=str, default='cfgs/single_client.ini')
parser.add_argument('--server', type=str, default=None)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cfg', type=str, default='cfgs/client0.ini')
parser.add_argument('--fusion', action='store_true')
parser.add_argument('--wandb_name', type=str, default=None) # delete later


args = parser.parse_args()

def main():

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.wandb_name == 'test':
        wandb.init(mode="disabled")
    else:
        wandb.init(project='madnet2-adatper',
                   name=args.wandb_name)

    with open(args.nodelist) as f:
        clients_file = [s.strip() for s in f.readlines() ]
        clients_ids = range(len(clients_file))

    run_file = args.cfg

    threads = [StereoClient(i,args,j) for i,j in zip(clients_file,clients_ids)]

    for i in range(len(threads)):
        threads[i].start()
    
if __name__ == '__main__':
   main()
