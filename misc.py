# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import argparse
import json
import logging
from copy import deepcopy

# from experiments import generate_experiment_cfgs
# from mmcv import Config, get_logger
from prettytable import PrettyTable

# from mmseg.models import build_segmentor


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def count_parameters(model, logger=None):
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, human_format(param)])
        total_params += param
    print(table)
    print(f'Total Trainable Params: {human_format(total_params)}')
    if logger:
        logger.info(table)
        logger.info(f'Total Trainable Params: {human_format(total_params)}')
    return total_params


def count_all_parameters(model, logger=None):
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, human_format(param)])
        total_params += param
    print(table)
    print(f'Total Params: {human_format(total_params)}')
    if logger:
        logger.info(table)
        logger.info(f'Total Params: {human_format(total_params)}')
    return total_params

