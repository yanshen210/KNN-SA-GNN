#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_graph_utils import train_utils


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # basic parameters
    parser.add_argument('--model_name', type=str, default='GAT', help='the name of the model')
    parser.add_argument('--sample_length', type=int, default=10000, help='batchsize of the training process')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/test', help='the directory to save the model')
    parser.add_argument('--fault_num', type=int, default=5, help='number of fault types')
    parser.add_argument('--sample_size', type=int, default=200, help='the number of samples for each fault type')
    parser.add_argument('--train_sample', type=int, default=10, help='the number of train samples for each fault type')
    parser.add_argument('--train_num', type=int, default=1, help='the number of train samples ')
    parser.add_argument('--data_save', type=str, default='data_C_k3', help='choose the data that you want')
    parser.add_argument('--my_way', type=str, default='KS', help='Whether add my way')
    parser.add_argument('--k_value', type=int, default=3, help='the k value of KNN')


    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='200,300', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--resume', type=str, default='', help='the directory of the resume training model')
    parser.add_argument('--max_epoch', type=int, default=400, help='max number of epoch')
    args = parser.parse_args()
    return args






if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + args.my_way + '_' + datetime.strftime(datetime.now(), '%H%M%S')
    # datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args)
    trainer.setup()
    trainer.train()



