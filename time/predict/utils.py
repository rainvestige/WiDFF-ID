# -*- coding: utf-8 -*-
import math
import argparse
import random

import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import seaborn as sn

def get_dataloader(mat_filename, data_name, label_name,
                   batch_size, shuffle, percent,
                   one_antenna=True, tx_rx_path=8,
                   num_classes=30, id_list=None, height_range=None, use_rnn=False):
    """Get the pytorch dataloader

    Args:
        mat_filename: filename of the .mat, the file contains data and label
                      matrix
        data_name: the variable name of data
        label_name: the variable name of label
        batch_size: set the batch size of the dataloader
        shuffle: set whether the dataset need to be shuffled
        percent: select specific percentage samples over every 500 samples(5sec)
        one_antenna: if choose one pair of antenna
        tx_rx_path: transceiver antenna pair
        height_range: tuple (start, end)
    Returns:
        pytorch dataloader
    """
    data_and_label = sio.loadmat(mat_filename)
    data = data_and_label[data_name] # N x 30 or N x 30 x 3 x 3
    label = data_and_label[label_name] # N x 6
    # the 1st column of label matrix is the index of personID, the latter are
    # biometrics
    label[:, 0] = label[:, 0] - 1 # [1-num_classes] -> [0-(num_classes-1)]

    if one_antenna: data = _select_antenna(data, tx_rx_path)

    if id_list: data, label = _select_person(data, label, id_list)
    if height_range: data, label = _select_height(data, label, height_range)
    data, label = _filter_data(data, label, percent, num_classes=num_classes)

    # get the dataloader
    num_samples = len(data)
    data = ( # N x 30 x 1 x 1 or N x 30 x 3 x 3
        torch.from_numpy(data)
        .type(torch.FloatTensor)
    )
    # N x 30 -> N x 30 x 1 x 1
    if data.ndim == 2:
        data = data.view(num_samples, 30, 1, 1)
    if use_rnn:
        data = data.reshape(num_samples, 30, 1) # adjust to RNN
    label = torch.from_numpy(label).type(torch.FloatTensor)
    dataset = TensorDataset(data, label)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle)

    return dataloader

def _select_person(data, label, id_list):
    r"""Select people out of all 42 peoples by `id_list`
    """
    # do not performing selection
    if id_list is None:
        return data, label

    boolean_array = np.zeros_like(label[:, 0], dtype=bool)
    for i in id_list:
        boolean_array = boolean_array | (label[:, 0] == i)

    label = label[boolean_array]
    # reallocate the person id to [0, len(id_list)-1]
    id_list.sort() # make sure id will not conflict
    for idx, pid in enumerate(id_list):
        tmp = label[label[:, 0] == pid]
        tmp[:, 0] = idx
        label[label[:, 0] == pid] = tmp

    return data[boolean_array], label

def _select_height(data, label, height_range):
    r"""Select data out of all 42 peoples by `height_range`

    Args:
        height_range: set the height range [start, end)
    """
    if height_range is None:
        return data, label

    (start, end) = height_range
    boolean_array = np.zeros_like(label[:, 0], dtype=bool)
    boolean_array = boolean_array | (
        (start <= label[:, 1]) & (label[:, 1] < end))
    # select the samples in height range
    label = label[boolean_array]
    # delete the height column
    label  = np.delete(label, 1, 1)
    id_list = np.unique(label[:, 0])
    print('id list {} for selected height range {}'.format(id_list,
          height_range))
    # reallocate the person id to [0, len(id_list)-1]
    for idx, pid in enumerate(id_list):
        tmp = label[label[:, 0] == pid]
        tmp[:, 0] = idx
        label[label[:, 0] == pid] = tmp

    return data[boolean_array], label


def _filter_data(data, label, percent=1, num_classes=13):
    """Filter samples from (data, label) according to percent.

    Args:
        data:
        label:
        percent:
        num_classes: the 1st column of label matrix is the index of personID,
            num_classes is the number of persons.

    """
    assert 0 < percent <= 1, "percent invalid"
    if math.isclose(percent, 1):
        return data, label

    num_samples = len(data)
    shape_ = list(data.shape)
    shape_[0] = 0
    ret_data = np.empty(shape_, float) # 0 x 30 x 3 x 3 or 0 x 30
    shape_ = list(label.shape)
    shape_[0] = 0
    ret_label = np.empty(shape_, float) # 0 x 6

    # iterate over every person's data
    for i in range(num_classes):
        filtered_data = data[label[:, 0] == i] # M x 30 x 3 x 3
        filtered_label = label[label[:, 0] == i] # M x 6
        num_data = len(filtered_data)
        # 500 = 100Hz * 5sec
        # simulate recording 5s csi samples of person
        k = num_data // 500
        for j in range(k):
            # (percent*500) x 30
            start_idx = int(j * 500)
            end_idx = int((j + percent) * 500)
            tmp_data = filtered_data[start_idx: end_idx, :] # 500 x 30 x 3 x3
            tmp_label = filtered_label[start_idx: end_idx, :] # 500 x 6
            ret_data = np.vstack((ret_data, tmp_data))
            ret_label = np.vstack((ret_label, tmp_label))

    return ret_data, ret_label

def _select_sensitive_antenna(data, tx_rx_path):
    r"""Select the transceiver with most sensitive antenna

    Args:
        data: [N x 30 x Ntx x Nrx] shape ndarray

    Return:
        the transceiver antennas with largest variance value
    """
    assert data.ndim == 4, 'number of dimension of array does not match'
    shape_ = data.shape
    assert 0 <= tx_rx_path < shape_[2]*shape_[3], 'error transceiver path select'
    var_ = np.var(data, (0, 1)) # [Ntx, Nrx]
    flatten_idx_array = np.argsort(var_, axis=None) # (axis=None->flatten)
    flatten_idx = flatten_idx_array[tx_rx_path] # incremental order
    if tx_rx_path == 1:
        flatten_idx = 8
    if tx_rx_path == 2:
        flatten_idx = 2
    print('Select {}th transceiver path'.format(flatten_idx))
    row_idx = flatten_idx // shape_[3]
    col_idx = flatten_idx % shape_[3]
    return data[:, :, row_idx, col_idx]

def _select_antenna(data, ant_index):
    """Select one antenna data

    Args:
        data: [N x 30 x Ntx x Nrx] shape ndarray.
        ant_index: antenna index. (valid value [0,8])

    Return:
        one antenna data
    """
    # Convert antenna index to [row, col] subscripts.
    shape = data.shape
    row_idx = ant_index // shape[3]
    col_idx = ant_index % shape[3]
    return data[:, :, row_idx, col_idx]

def parse_arguments_():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', help='training batch size',
                        type=int, default=20)
    parser.add_argument('-n', '--num_epochs', help='the number of epochs',
                        type=int, default=20)
    parser.add_argument('-w', '--weight_decay', help='L2 regularizer weight decay',
                        type=float, default=5e-3)
    parser.add_argument('-p', '--percent', help='the percent of every 500 samples',
                        type=float, default=1.0)
    parser.add_argument('-o', '--one_antenna', help='use one transceiver antenna',
                        action='store_true')
    parser.add_argument('--tx_rx_path', help='transceiver path selection',
                        type=int, default=8)
    parser.add_argument('-i', '--index', help='selected_id index',
                        type=int)
    parser.add_argument('--num_classes', help='number of classes',
                        type=int)
    args = parser.parse_args()
    return args

def humanid_eval(net, dataloader, is_test=False):
    r"""Evaluating the human ID performance

    Args:
        net: the network model
        dataloader: the data for evaluating

    Returns:
        accuracy, all predict label
    """
    correct = 0
    num_instances = len(dataloader.dataset)
    # for now only store human id
    all_pred_label = np.empty(0)

    for (samples, labels) in tqdm(dataloader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())

            predict_label = net(samplesV) # (humanid, biometric)
            predict_label = list(predict_label)
            if labelsV.ndim == 1 or predict_label[0].ndim == 1:
                labelsV = labelsV.unsqueeze(0)
                predict_label = [pred.unsqueeze(0) for pred in predict_label]

            pred = predict_label[0].data.max(1)[1] # max(1) -> (values, indices)
            correct += pred.eq(labelsV[:, 0].data.long()).sum()
            if is_test:
                all_pred_label = np.concatenate(
                    (all_pred_label, pred.cpu().detach().numpy()), axis=0)

            ##plot the feature map
            #filename_prefix = '../figures/densenet_maxvar/deconv_feature_map'
            #if True:
            #    #print(predict_label[2].shape)
            #    for j in range(6):
            #        img = predict_label[2][2, j, :, :]
            #        img = img.cpu().detach().numpy()
            #        #plt.imshow(img, interpolation='nearest')
            #        #plt.matshow(img)
            #        sn.heatmap(img, cmap='viridis')
            #        plt.axis('off')
            #        #plt.axes([0, 0, 1, 1])
            #        plt.imsave(
            #            filename_prefix + str(j) + '.png', img,
            #            format='png')
            #        #plt.title('Person ID: ' + str(pred[1]))
            #        plt.show()
            #    break
    acc = float(100.0 * correct / num_instances)
    if is_test:
        assert len(all_pred_label) == num_instances, (
               'the number of predicted label does not match')
        return acc, all_pred_label
    else:
        return acc

def biometric_eval(net, dataloader, num_classes=30):
    r"""

    Returns:
        the mean average estimation error for all biometrics [1, 5]
    """
    # compute the average estimation error
    bio_ae = torch.zeros(num_classes, 5).cuda() # 13 x 5
    # compute the square deviation
    bio_sd = torch.zeros(num_classes, 5).cuda() # 13 x 5
    total_label = dataloader.dataset[:][1].cpu().detach().numpy()
    num_person = [(total_label[:, 0] == i).sum() for i in range(num_classes)]

    for (samples, labels) in tqdm(dataloader):
        samplesV = Variable(samples.cuda()) # batch_size x 30 x 1 x1
        labelsV = Variable(labels.cuda()) # batch_size x 6

        predict_label = net(samplesV) # (humanid, biometric)
        predict_label = list(predict_label)
        if labelsV.ndim == 1 or predict_label[0].ndim == 1:
            labelsV = labelsV.unsqueeze(0)
            predict_label = [pred.unsqueeze(0) for pred in predict_label]

        pred_bio = predict_label[1].data # batch_size x 5
        true_bio = labelsV[:, 1:] # batch_size x 5
        true_id = labelsV[:, 0] # batch_size x 1
        for i in range(num_classes):
            bio_ae[i, :] = bio_ae[i, :] + (
                (true_bio[true_id == i] - pred_bio[true_id == i]).abs().sum(0) /
                num_person[i])
            bio_sd[i, :] = bio_sd[i, :] + (
                (pred_bio[true_id == i] - true_bio[true_id == i]).pow(2).sum(0) /
                num_person[i])
    # compute the mean average estimation error
    bio_mae = bio_ae.mean(0).cpu().detach().numpy()
    bio_msd = bio_sd.mean(0).cpu().detach().numpy()
    return bio_mae, bio_msd

def set_selected_id(num_rows, num_cols):
    r"""Set selected id array of shape [num_rows x num_cols]

    Select `num_cols` people out of all 42 people `num_rows` times

    Args:
        num_rows: the times performing selection
        num_cols: the number of people selected
    """
    selected_id = np.empty((0, num_cols), dtype=int)
    id_list = [_ for _ in range(42)]
    ## first 2 id entry is specified, because i have done experiments on this
    ## selected id
    #removed_id = [11, 12, 17, 18, 19, 20, 22, 23, 24, 28, 35, 41]
    #id_entry = [_ for _ in id_list if _ not in removed_id]
    #id_entry.sort()
    #id_entry = np.asarray(id_entry, dtype=int)
    #selected_id = np.vstack((selected_id, id_entry))
    #id_entry = np.array(
    #    [0, 2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 17, 18, 19, 20, 21,
    #     22, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38], dtype=int)
    #selected_id = np.vstack((selected_id, id_entry))

    # Do not consider the repeated entry
    for _ in range(num_rows):
        id_entry = random.sample(id_list, num_cols)
        id_entry.sort()
        id_entry = np.asarray(id_entry, dtype=int)
        selected_id = np.vstack((selected_id, id_entry))
    print(selected_id)

    # save to `.mat` file
    dirname = '../data/20201120/'
    filename = dirname + 'select_' + str(num_cols) + '.mat'
    sio.savemat(filename, {'selected_id': selected_id})

def get_selected_id(filename, idx):
    data = sio.loadmat(filename)
    selected_id = data['selected_id']
    return (selected_id[idx, :]).tolist()


if __name__ == '__main__':
    for i in range(21, 25, 1):
        if i % 5 == 0:
            continue
        set_selected_id(5, i)
    #set_selected_id(5, 35)
    #print(get_selected_id('../data/20201120/select_30.mat', 2))
    #filename = '../data/20201120/dim_aug_train.mat'
    #dataset = sio.loadmat(filename)
    #data = dataset['aug_train_data']
    #label = dataset['aug_train_label']
    #label[:, 0] = label[:, 0] - 1

    #height_range = (180, 200)
    #if height_range: data, label = _select_height(data, label, height_range)
    #print('After selection, data shape is {}, label shape is{}'.format(
    #      data.shape, label.shape))

