#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import random

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label) 
        return image, label # CIFAR等数据集dataset已经包含了totensor操作


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, gradient_store, last_update):
        '''
        Use DGC to cut small gradient change
        '''
        # Set mode to train model
        model.train()
        epoch_loss = []
        if gradient_store == None: # 第一轮训练，用于更新的梯度值为0
            gradient_store = model.state_dict()
            for key,val in gradient_store.items():
                gradient_store[key] = val*0.0
        if last_update == None: # 第一轮训练，用于动量累计的梯度值为0
            last_update = model.state_dict()
            for key,val in last_update.items():
                last_update[key] = val*0.0

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        model_dict_ori = copy.deepcopy(model.state_dict())
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)     # clip gradient
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradient_update = calculate_gradient(model_dict_ori,model.state_dict())
        gradient_update = merge_gradient(gradient_update,last_update,0.5)   # 考虑动量的梯度更新 u_k,t
        last_update_out = gradient_update
        gradient_update = merge_gradient(gradient_store,gradient_update)    # G_t = G_t + G_(t-1)
        # sparse_rates = [0.75,0.9375,0.9843,0.99]    # 热身阶段的稀疏率
        sparse_rates = [0.75,0.75,0.75,0.80]
        if global_round<4:
            sparse_rate = sparse_rates[global_round]
        else:
            sparse_rate = sparse_rates[-1]
        gradient_update, gradient_store = self.sparse_gradient_mask(gradient_update,sparse_rate)    # 计算G_t的主要梯度和次要梯度
        del model
        return gradient_update,gradient_store,last_update_out,sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss
    
    def sparse_gradient_mask(self,model:dict,sparse_rate:float=0.75):
        '''
        Calculate all masked G[j],j=0,1,2... at communicate epoch T for user idx
        Thresold is set as the absolute value of top (1-k)% in G[j], k% is sparse_rate
        '''
        mask_dict = copy.deepcopy(model)
        mask_dict_inv = copy.deepcopy(model)
        for key,tensors in model.items():
            flat_tensors = torch.flatten(tensors)
            # 目前取k=75
            tensor_len = len(flat_tensors)
            sample_num = min(tensor_len,80)
            index = torch.LongTensor(random.sample(range(tensor_len),sample_num))
            samples = flat_tensors[index]
            thr = torch.topk(torch.abs(samples),k=int(np.ceil(sample_num*(1-sparse_rate))))[0][-1]
            # torch.tensor(1/np.sqrt(self.args.num_users))
            mask_j = mask_dict[key]
            mask_j[torch.abs(mask_j)<thr] = 0
            mask_dict[key] = mask_j

            mask_j_inv = mask_dict_inv[key]
            mask_j_inv[torch.abs(mask_j_inv)>thr] = 0
            mask_dict_inv[key] = mask_j_inv
        # print('Ori upload pack norm:',torch.norm(tensors))  # debug
        # print('sparse upload pack norm:',torch.norm(mask_j))    # debug
        return mask_dict, mask_dict_inv
    
     
def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def calculate_gradient(model_ori:dict,model:dict)->dict:
    '''
    Return the gradient update
    '''
    if model_ori.keys() != model.keys():
        raise RuntimeError('Models do not has same structure')
    gradient = {}
    for key,_ in model.items():
        gradient[key] = model[key] - model_ori[key]

    return gradient

def merge_gradient(gradient1:dict,gradient2:dict,momentum = None)->dict:
    '''
    Return plus of two gradient. Momentum will scale the later gradient if it exists
    '''
    if gradient1.keys() != gradient2.keys():
        raise RuntimeError('Models do not has same structure')
    gradient = {}
    if momentum != None:
        for key,_ in gradient1.items():
            gradient[key] = gradient1[key] + gradient2[key] * momentum
    for key,_ in gradient1.items():
        gradient[key] = gradient1[key] + gradient2[key]

    return gradient



