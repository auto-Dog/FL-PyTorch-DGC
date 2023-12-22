#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid

class GradientList:
    '''
    存储在内存里的模型参数列表
    ---
    example:
    g_list = GradientList(5)    # g_list可以像正常列表一样使用。列表里的模型都将被存入内存  

    g_list[2] = model.state_dict()  # 赋值，并将显存中的模型参数转移到内存 

    read = g_list[2]    # 读取，将内存中的模型参数转移到显存  
    '''  
    def __init__(self,num):
        self.storage = [None for i in range(num)]
        self.length = num
    def __getitem__(self,index):
        '''Output a dict of tensor to GPU'''
        if self.storage[index] == None:
            return None
        new_dict = {}
        source_dict = self.storage[index]
        for key,val in source_dict.items():
            new_dict[key] = val.cuda()
        return new_dict
    def __setitem__(self,index,value):
        '''Input a dict of tensor from GPU'''
        new_dict = {}
        for key,val in value.items():
            new_dict[key] = val.detach().cpu()
        self.storage[index] = new_dict

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '/kaggle/input/cifar10-python'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir,train=True, 
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir,train=False, 
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w:list):
    """
    Returns the average of the weights.
    """
    w_avg = w[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights_gradient(w:dict, gradient_packs:list):
    """
    Returns the weights with gradient updates.
    """
    w_avg = gradient_packs[0]
    for key in w_avg.keys():
        for i in range(1, len(gradient_packs)):
            w_avg[key] += gradient_packs[i][key]
        w_avg[key] = torch.div(w_avg[key], len(gradient_packs))
        w_avg[key] += w[key]
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
