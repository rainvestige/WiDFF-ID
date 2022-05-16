# -*- coding: utf-8 -*-
import time

import torch
import numpy as np
import pandas as pd
import scipy.io as sio

from torch.autograd import Variable

def main():
    # load network model
    net = torch.load('./aug_pca8.pkl')
    net = net.cuda()
    # only evaluate
    net.eval()

    num_iter = 100
    total_time = 0
    for i in range(num_iter):
        time_start = time.clock()
        # Prepare data, nSample x nChannel x width x height
        test_fname = 'pca_data.mat'
        dataset = sio.loadmat(test_fname)
        data = dataset['pca_data']
        data = torch.from_numpy(data).type(torch.FloatTensor)
        num_samples = len(data)
        data = data.view(num_samples, 30, 1, 1)

        with torch.no_grad():
            samples = Variable(data.cuda())
            predict_label = net(samples)
            predict_label = predict_label[0].data.max(1)[1]
            bincount = predict_label.bincount()
            prob = bincount.max(-1)[0].item() / num_samples
            predicted_id = bincount.max(-1)[1].item() + 1
            if prob > 0.8:
                print("Identified as No.{}, the probability is "
                      "{}".format(predicted_id, prob))
            else:
                print("Illegal person")
        elapsed_time = (time.clock() - time_start)
        total_time += elapsed_time
    print("The time taken to identify a unknown person: {:4.6f}ms".format(
        total_time / num_iter * 1000))




if __name__ == '__main__':
    main()
