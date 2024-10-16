import threading
import configparser
import random

from clients import StereoClient
from server import StereoServer

import argparse
import torch
import numpy as np

import os
import logging
import datetime
import wandb

WORK_DIR = './work_dirs/exp1-base'

DEBUG = True

parser = argparse.ArgumentParser(description='FedStereo')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--nodelist', type=str, default='cfgs/single_client.ini')
parser.add_argument('--server', type=str, default=None)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--visualize', type=bool, default=True)
args = parser.parse_args()

def main():

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.nodelist) as f:
        clients_file = [s.strip() for s in f.readlines() ]
        clients_ids = range(len(clients_file))

    server = None
    print('Clients:\n%s'%str(clients_file))
    if args.server is not None:
        print('Server:\n%s'%args.server)
        server = StereoServer(args.server,args)

    if len(clients_file) == 1: #!DEBUG
        a=1
        assert os.path.isdir(WORK_DIR)
        train_serial = str(datetime.datetime.now())
        if DEBUG:
            train_serial = f"debug" #_{train_serial}"
        LOG_DIR = os.path.join(WORK_DIR, train_serial)
        os.makedirs(LOG_DIR, exist_ok=True)

        root_logger = logging.getLogger(name='')
        root_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(os.path.join(LOG_DIR, 'train.log'))
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    threads = [StereoClient(i, args, j,
                            server=server,
                            logger=root_logger,
                            exp_dir=LOG_DIR,
                            visualize=args.visualize) for i,j in zip(clients_file,clients_ids)]

    a=1
    for t in threads:
        if t.listener and server is not None:
            server.link_listening_client(t)

        if t.sender and server is not None:
            server.link_sending_client(t)

    # Starting threads
    if server is not None:
        server.start()
    for i in range(len(threads)):
        threads[i].start()

if __name__ == '__main__':
   main()
