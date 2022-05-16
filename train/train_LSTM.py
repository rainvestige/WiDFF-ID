# -* coding:utf-8 -*

import argparse
import os
import math
import time

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from utils import get_dataloader, parse_arguments_, humanid_eval
from utils import get_selected_id
from model.res_net_use_this import ResNet, ResidualBlock, Bottleneck
from model.use_densenet import CSIDenseNet
from model.RNN import Net
from model.LSTM import lstm

################################################################################
# command line options
args = parse_arguments_()
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = 0.001
weight_decay = args.weight_decay
percent = args.percent
one_antenna = args.one_antenna
tx_rx_path = args.tx_rx_path
index = args.index
NUM_CLASSES = args.num_classes

COMMON_DIR = ('LSTM/24-6_12_36/' + str(NUM_CLASSES) + '_' +
            str(index+1) + 'th/')
MODEL_SAVEDIR = '../weights/' + COMMON_DIR
MODEL_SAVENAME = (MODEL_SAVEDIR + 'aug_pca' + str(tx_rx_path) +
                  '.pkl')
ACC_LOSS_SAVEDIR = ('../data/result/' + COMMON_DIR + 'aug_pca' +
                    str(tx_rx_path) + '/')
################################################################################
id_list = None
height_range = None
train_data_loader = get_dataloader(
    'dim_aug_train_.mat',
    'aug_train_data', 'aug_train_label',
    batch_size, shuffle=True, percent=percent,
    one_antenna=one_antenna, tx_rx_path=tx_rx_path,
    num_classes=NUM_CLASSES, id_list=id_list, height_range=height_range, use_rnn=True)
test_data_loader = get_dataloader(
    'dim_filter_test_.mat',
    'filter_test_data', 'filter_test_label',
    batch_size, shuffle=False, percent=1,
    one_antenna=one_antenna, tx_rx_path=tx_rx_path,
    num_classes=NUM_CLASSES, id_list=id_list, height_range=height_range, use_rnn=True)

num_train_instances = len(train_data_loader.dataset)
num_test_instances = len(test_data_loader.dataset)

################################################################################
net = lstm(input_size=1, hidden_size=100, num_layers=1)

net = net.cuda()

criterion1 = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[i for i in range(3, num_epochs, 3)],
    gamma=0.5)

# record the acc for plotting, one acc per epoch
acc = {'trainacc': [], 'testacc': []}
bestacc = 0
loss_record = {'loss0': []}
for epoch in range(num_epochs):
    print('Epoch:', epoch, 'Weight Decay:', weight_decay, 'lr:', learning_rate,
          'percent:', percent)
    net.train()

    # trained_num = 0
    for (samples, labels) in tqdm(train_data_loader):

        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label = net(samplesV)
        predict_label = list(predict_label)
        # batch size lead to only on sample
        if labelsV.ndim == 1 or predict_label[0].ndim == 1:
            labelsV = labelsV.unsqueeze(0)
            predict_label[0] = predict_label[0].unsqueeze(0)
        lossC = criterion1(predict_label[0], labelsV[:, 0].type(torch.LongTensor).cuda())

        loss_record['loss0'].append(lossC.cpu().detach().numpy())

        loss = lossC
        loss.backward()
        optimizer.step()
    scheduler.step()

    # evaluating human id performance
    net.eval()
    tmp_acc = humanid_eval(net, train_data_loader)
    print('Training Accuracy: {:.3f}'.format(tmp_acc))
    trainacc = str(tmp_acc)[0:6]

    tmp_acc = humanid_eval(net, test_data_loader)
    print('Test Accuracy: {:.3f}'.format(tmp_acc))
    testacc = str(tmp_acc)[0:6]

    acc['trainacc'].append(float(trainacc))
    acc['testacc'].append(float(testacc))
    if not os.path.exists(MODEL_SAVEDIR): os.makedirs(MODEL_SAVEDIR)
    if float(testacc) > bestacc:
        bestacc = float(testacc)
        torch.save(net, MODEL_SAVENAME)
################################################################################
if not os.path.exists(ACC_LOSS_SAVEDIR): os.makedirs(ACC_LOSS_SAVEDIR)
# save the loss `.mat` file
saved_fn = (ACC_LOSS_SAVEDIR + 'training_loss_percent' + str(percent) + '.mat')
sio.savemat(saved_fn, loss_record)
# save the accuracy `.mat` file
saved_fn = (ACC_LOSS_SAVEDIR + 'acc_nEpochs' + str(num_epochs) +
            'weightDecay' + str(weight_decay) +
            'percent' + str(percent) + '.mat')
sio.savemat(saved_fn, acc)

