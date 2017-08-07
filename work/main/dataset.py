import os
import os.path
import errno
import numpy as np
import sys

import torch

import torch.utils.data as data

import quandl
quandl.ApiConfig.api_key = 'h2ho2wNxjkegKK9gtdDu'

def seq(num_data, num_days, std=1, shift_range=1e3, scale_range=1e2):
    data = []
    labels = []
    for i in range(num_data):
        if np.random.randint(2):
            seq = np.arange(num_days)+np.random.normal(0, std, num_days)+np.random.randint(shift_range)
            seq *= (np.random.randint(scale_range)+1)
            data.append(seq)
            labels.append(1.0)
        else:
            seq = np.arange(num_days-1, -1, -1)+np.random.normal(0, std, num_days)+np.random.randint(shift_range)
            seq *= (np.random.randint(scale_range)+1)
            data.append(seq)
            labels.append(-1.0)
    data_arr = np.array(data)
    # Normalize the data
    mean = data_arr.mean(axis=0)
    std = data_arr.std(axis=0)
    data_arr -= mean
    data_arr /= std
    labels_arr = np.array(labels)
    return data_arr, labels_arr

class DummyDataset(data.Dataset):
    """
    """

    def __init__(self, train=True, num_train=1000, num_test=1000, num_days=100):
        data = quandl.get_table('WIKI/PRICES', ticker='FB')

        self.train = train  # training set or test set
        self.num_days = num_days
        self.num_train = num_train
        self.num_test = num_test

        # now load the picked numpy arrays
        if self.train:
            self.train_data, self.train_labels = seq(self.num_train, self.num_days)
        else:
            self.test_data, self.test_labels = seq(self.num_test, self.num_days)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            data, target = self.train_data[index], self.train_labels[index]
        else:
            data, target = self.test_data[index], self.test_labels[index]

        data_tensor = torch.from_numpy(data).double()
        target_tensor = torch.from_numpy(np.array([target]))

        return data_tensor, target_tensor

    def __len__(self):
        if self.train:
            return self.num_train
        else:
            return self.num_test
