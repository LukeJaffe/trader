#!/usr/bin/env python3

import numpy as np

data_file = 'wiki_data_100_50_7.npy'
label_file = 'wiki_labels_100_50_7.npy'

train_data_file = 'wiki_train_data_100_50_7.npy'
train_label_file = 'wiki_train_labels_100_50_7.npy'

test_data_file = 'wiki_test_data_100_50_7.npy'
test_label_file = 'wiki_test_labels_100_50_7.npy'

data_arr = np.load(data_file)
label_arr = np.load(label_file)

num_test = data_arr.shape[0]//10
train_data_arr = data_arr[:-num_test]
train_label_arr = label_arr[:-num_test]
test_data_arr = data_arr[-num_test:]
test_label_arr = label_arr[-num_test:]

np.save(train_data_file, train_data_arr)
np.save(test_data_file, test_data_arr)
np.save(train_label_file, train_label_arr)
np.save(test_label_file, test_label_arr)
