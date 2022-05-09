#!/usr/bin/python
# -*- coding:utf-8 -*-


import logging
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
import random
import numpy as np

from utils.Set_division import Set_division

class train_utils(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        self.datasets_train, self.datasets_test = Set_division(args).train_test()


        # Define the model
        feature = int(args.sample_length/2)

        self.model = getattr(models, args.model_name)(feature=feature,out_channel=args.fault_num)


        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.resume:
            suffix = args.resume.rsplit('.', 1)[-1]
            if suffix == 'tar':
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suffix == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):

        args = self.args

        # 定义训练集标签样本和无标签样本
        samples = np.array(range(args.train_sample*args.fault_num))
        train, train_mask, label_mask = [], [], []
        edge_train = self.datasets_train.edge_index

        for i in range(args.fault_num):
            sample = samples[i * args.train_sample:(i + 1) * args.train_sample]
            random.seed()
            random.shuffle(sample)
            train += torch.LongTensor(sample[:args.train_num])  # 真实标签样本
            train_mask += torch.LongTensor(sample[:args.train_num])  # 全部标签样本
            label_mask += torch.LongTensor(np.zeros(args.train_num) + i)  # 全部标签
            for j in sample[:args.train_num]:
                for k in range(args.k_value):
                    # if edge_train[1][args.k_value*j+k] not in train:
                    train_mask += edge_train[1][args.k_value*j+k].unsqueeze(0)  # 全部标签样本
                    label_mask += torch.LongTensor(np.zeros(1) + i)  # 全部标签
        train_mask = torch.LongTensor(train_mask)
        label_mask = torch.LongTensor(label_mask)
        idx_train = torch.LongTensor(train)

        # 计算伪标签准确率
        labels_true = self.datasets_train.y[train_mask]
        true = torch.eq(label_mask, labels_true).float().sum().item()
        acc = (true-args.train_num*args.fault_num)/(args.k_value*args.train_num*args.fault_num)
        logging.info('\n 伪标签准确率：{:.4f} \n'.format(acc))

        # 数据写入cuda
        inputs_train = self.datasets_train.to(self.device)
        inputs_test = self.datasets_test.to(self.device)
        idx_train = idx_train.to(self.device)
        train_mask = train_mask.to(self.device)
        label_mask = label_mask.to(self.device)

        labels_train = inputs_train.y
        labels_test = inputs_test.y
        time_start = time.time()


        for epoch in range(self.start_epoch, args.max_epoch):


            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))



            # Train
            if epoch < args.max_epoch-1:
                self.model.train()
                with torch.set_grad_enabled(True):
                    # forward
                    logits = self.model(inputs_train)  # 输入训练集

                    if args.my_way == 'KS':
                        loss = self.criterion(logits[train_mask], label_mask)
                        pred = logits[train_mask].argmax(dim=1)
                        correct = torch.eq(pred, labels_train[train_mask]).float().sum().item()
                        epoch_loss = loss.item()
                        epoch_acc = correct / (len(label_mask))
                    else:
                        loss = self.criterion(logits[idx_train], labels_train[idx_train])
                        pred = logits[idx_train].argmax(dim=1)
                        correct = torch.eq(pred, labels_train[idx_train]).float().sum().item()
                        epoch_loss = loss.item()
                        epoch_acc = correct / (args.train_num * args.fault_num)

                    # backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                logging.info('Epoch: {} train-Loss: {:.6f} train-Acc: {:.4f}'.format(
                    epoch, epoch_loss, epoch_acc))


            else:
                # The last train
                self.model.train()
                with torch.set_grad_enabled(True):
                    # forward
                    logits = self.model(inputs_train)  # 输入训练集

                    if args.my_way == 'KS':
                        loss = self.criterion(logits[train_mask], label_mask)
                        pred = logits[train_mask].argmax(dim=1)
                        correct = torch.eq(pred, labels_train[train_mask]).float().sum().item()
                        epoch_loss = loss.item()
                        epoch_acc = correct / (len(label_mask))
                    else:
                        loss = self.criterion(logits[idx_train], labels_train[idx_train])
                        pred = logits[idx_train].argmax(dim=1)
                        correct = torch.eq(pred, labels_train[idx_train]).float().sum().item()
                        epoch_loss = loss.item()
                        epoch_acc = correct / (args.train_num * args.fault_num)

                    # backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                logging.info('Epoch: {} train-Loss: {:.6f} train-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, epoch_loss, epoch_acc, time.time() - time_start))


                # The test
                time_start = time.time()
                self.model.eval()
                logits = self.model(inputs_test)  # 输入测试集
                loss = self.criterion(logits, labels_test)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, labels_test).float().sum().item()
                epoch_loss = loss.item()
                epoch_acc = correct / ((args.sample_size - args.train_sample) * args.fault_num)

                logging.info('Epoch: {} test-Loss: {:.6f} test-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, epoch_loss, epoch_acc, time.time()-time_start))










