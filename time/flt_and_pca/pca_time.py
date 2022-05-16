#! -*- coding: utf-8 -*-
import time

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


def load_dataset(filename, data_name, label_name):
    dataset = sio.loadmat(filename)
    data = dataset[data_name]
    label = dataset[label_name]
    return [data, label]

def get_pca(data, n_components=30):
    r"""Reduce the dimensionality using PCA

    Args:
        data: the input data for fitting PCA
        n_components: the number of components

    Returns:
        PCA instance
    """
    data = data.reshape(data.shape[0], -1) # [N,30,3,3] -> [N,30*3*3]
    # check the data shape
    assert data.ndim == 2, "the number of dim of data for pca must be 2"
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(data)
    return pca

def apply_pca(dirname, filename, dataname, labelname, pca_matrix=None):
    r"""Apply the PCA on data loaded from specific filename

    Args:
        pca_matrix: the SVD matrix

    Returns:
        pca_matrix
    """
    # performing PCA on training dataset
    print('Performing PCA on data loaded from {}'.format(filename))
    [data, label] = load_dataset(
        dirname + filename, dataname, labelname)
    print('shape of data before dim reduction: {}'
          .format(data.shape))

    if not pca_matrix: pca_matrix = get_pca(data)

    data = data.reshape(data.shape[0], -1)
    reduced_data = pca_matrix.transform(data)
    print('shape of data after dim reduction: {}'
          .format(reduced_data.shape))
    sio.savemat(
        dirname+'dim_'+filename,
        {dataname: reduced_data, labelname: label})
    print('*'*80)
    return pca_matrix

def main():
    train_dirname = (
    '/home/public/b509/code/g19/xxy/projects/CSI-DenseNet/data/20201120/')
    filename = 'aug_train.mat'
    dataname = 'aug_train_data'
    labelname = 'aug_train_label'
    pca_matrix = apply_pca(train_dirname, filename, dataname, labelname,
                           pca_matrix=None)

    test_dirname = './'
    filename = 'filtered_data.mat'
    data_key = 'filtered_data'

    total_time = 0
    num_iterations = 100
    reduced_data = None
    for i in range(num_iterations):
        time_start = time.clock()
        dataset = sio.loadmat(test_dirname + filename)
        data = dataset[data_key]
        data = data.reshape(data.shape[0], -1)
        reduced_data = pca_matrix.transform(data)
        time_elapsed = (time.clock() - time_start)
        total_time += time_elapsed
    print("The time taken to perform a principal component analysis on 50 "
          "samples is: {:4.6f}ms".format(total_time / num_iterations * 1000))
    sio.savemat('pca_data.mat', {'pca_data': reduced_data})

if __name__ == '__main__':
    main()

