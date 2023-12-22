#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar


if __name__ == '__main__':
    args = args_parser()
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = CNNCifar(args=args)
            from torchvision import models
            global_model = models.resnet50(num_classes = 10)    # 和DGC实验保持一致
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []
    epoch_acc = []
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []
        global_model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if batch_idx % 50 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, batch_idx * len(images), len(trainloader.dataset),
            #         100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        
        global_model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)  
            loss = criterion(outputs, labels)
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        test_accuracy = correct/total
        epoch_loss.append(loss_avg)
        epoch_acc.append(test_accuracy)

        if (epoch%20 == 0 and epoch>0):
            torch.save(global_model.state_dict(),'./model_e{}_base.pth'.format(epoch))
        # print('\n Loss:{:4.3f} | Accuracy:{:4.3f}'.format(loss_avg,test_accuracy))

    np.savetxt('./save/acc_and_loss.csv',np.array([epoch_loss, epoch_acc]),delimiter=',')  # 输出结果用于绘图
    
    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
