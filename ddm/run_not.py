import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
import pdb
from ddm.alg.vcl import run_vcl, run_vcl_ibp
from ddm.alg.coreset import rand_from_batch, k_center
from ddm.alg.utils import plot
from copy import deepcopy

class NotMnistGenerator():
    def __init__(self):
        with open('ddm/data/notMNIST.pickle', 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        self.X_train = d['train_dataset'].reshape((-1, 28*28))
        self.train_label = d['train_labels']
        self.X_val = d['valid_dataset'].reshape((-1, 28*28))
        self.val_label = d['valid_labels']
        self.X_test = d['test_dataset'].reshape((-1, 28*28))
        self.test_label = d['test_labels']

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]

        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            val_0_id = np.where(self.val_label == self.sets_0[self.cur_iter])[0]
            val_1_id = np.where(self.val_label == self.sets_1[self.cur_iter])[0]
            next_x_val = np.vstack((self.X_val[val_0_id], self.X_val[val_1_id]))

            next_y_val = np.vstack((np.ones((val_0_id.shape[0], 1)), np.zeros((val_1_id.shape[0], 1))))
            next_y_val = np.hstack((next_y_val, 1 - next_y_val))
            self.cur_iter += 1
            return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val

if __name__ == "__main__":
    pass