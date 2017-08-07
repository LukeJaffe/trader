import os
import os.path
import errno
import numpy as np
import sys

import torch

import torch.utils.data as data

class WikiDataset(data.Dataset):
    """
    """

    train_data_file = '../data/wiki_train_data_100_50_7.npy'
    train_label_file = '../data/wiki_train_labels_100_50_7.npy'
    test_data_file = '../data/wiki_test_data_100_50_7.npy'
    test_label_file = '../data/wiki_test_labels_100_50_7.npy'

    def __init__(self, train=True, num_train=1000, num_test=10000, num_days=100,
        train_mean=None, train_std=None):
        self.train = train  # training set or test set
        self.num_days = num_days
        self.num_train = num_train
        self.num_test = num_test

        # now load the picked numpy arrays
        if self.train:
            self.train_data = np.load(self.train_data_file)[:self.num_train]
            self.train_labels = np.load(self.train_label_file)[:self.num_train]
            self.train_mean = self.train_data.mean(axis=0)
            self.train_std = self.train_data.std(axis=0)
            #self.train_data -= self.train_mean
            #self.train_data /= self.train_std
        else:
            self.test_data = np.load(self.test_data_file)[:self.num_test]
            self.test_labels = np.load(self.test_label_file)[:self.num_test]
            #self.test_data -= train_mean
            #self.test_data /= train_std

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
        target_tensor = torch.from_numpy(np.array([target])).double()

        return data_tensor, target_tensor

    def __len__(self):
        if self.train:
            return self.num_train
        else:
            return self.num_test
