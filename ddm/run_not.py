import numpy as np
import tensorflow as tf
import pickle
import pdb
import os.path
import sys
import argparse
sys.path.extend(['alg/'])
from cla_models_multihead import Vanilla_NN, MFVI_IBP_NN
from vcl import run_vcl
from utils import get_scores, concatenate_results
from copy import deepcopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class NotMnistGenerator():
    def __init__(self):
        with open('data/notMNIST.pickle', 'rb') as f:
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

    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')
    args = parser.parse_args()

    print('tag          = {!r}'.format(args.tag))

    seeds = [12, 13, 14, 15, 16]

    vcl_ibp_accs = np.zeros((len(seeds), 5, 5))
    vcl_h5_accs = np.zeros((len(seeds), 5, 5))
    vcl_h10_accs = np.zeros((len(seeds), 5, 5))
    vcl_h50_accs = np.zeros((len(seeds), 5, 5))

    # bayes_opt params
    alpha0 = 2.9
    beta0 = 1.0
    lambda_1 = 0.1
    lambda_2 = 1.0

    for i in range(len(seeds)):
        s = seeds[i]
        hidden_size = [100]
        batch_size = 128
        no_epochs = 1000
        ibp_samples = 10

        tf.set_random_seed(s)
        np.random.seed(1)

        ibp_acc = np.array([])

        coreset_size = 0
        data_gen = NotMnistGenerator()
        single_head = False
        in_dim, out_dim = data_gen.get_dims()
        x_testsets, y_testsets = [], []
        for task_id in range(data_gen.max_iter):

            tf.reset_default_graph()
            x_train, y_train, x_test, y_test, _, _ = data_gen.next_task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)

            # Set the readout head to train
            head = 0 if single_head else task_id
            bsize = x_train.shape[0] if (batch_size is None) else batch_size

            # Train network with maximum likelihood to initialize first model
            if task_id == 0:
                ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
                ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
                mf_weights = ml_model.get_weights()
                mf_variances = None
                mf_betas = None
                ml_model.close_session()

            # Train on non-coreset data
            # lambda_1 --> temp of the variational Concrete posterior
            # lambda_2 --> temp of the relaxed prior, for task != 0 this should be lambda_1!!!
            mf_model = MFVI_IBP_NN(in_dim, hidden_size, out_dim, x_train.shape[0], num_ibp_samples=ibp_samples,
                                   prev_means=mf_weights,
                                   prev_log_variances=mf_variances, prev_betas=mf_betas,
                                   alpha0=alpha0, beta0=beta0,
                                   learning_rate=0.0001,
                                   lambda_1=lambda_1,
                                   lambda_2=lambda_2 if task_id == 0 else lambda_1,
                                   no_pred_samples=100,
                                   name='ibp_not_split_mnist_run{0}_{1}'.format(i+1, args.tag))

            mf_model.train(x_train, y_train, head, no_epochs, bsize,
                           anneal_rate=0.0, min_temp=1.0)
            mf_weights, mf_variances, mf_betas = mf_model.get_weights()

            acc = get_scores(mf_model, x_testsets, y_testsets, single_head)
            ibp_acc = concatenate_results(acc, ibp_acc)
            mf_model.close_session()
        vcl_ibp_accs[i, :, :] = ibp_acc


        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [10]
        data_gen = NotMnistGenerator()
        vcl_result_h10 = run_vcl(hidden_size, no_epochs, data_gen,
                             lambda a: a, coreset_size, batch_size, single_head, val=True)
        vcl_h10_accs[i, :, :] = vcl_result_h10

        tf.reset_default_graph()
        hidden_size = [5]
        data_gen = NotMnistGenerator()
        vcl_result_h5 = run_vcl(hidden_size, no_epochs, data_gen,
                                lambda a: a, coreset_size, batch_size, single_head, val=True)
        vcl_h5_accs[i, :, :] = vcl_result_h5

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [50]
        data_gen = NotMnistGenerator()
        vcl_result_h50 = run_vcl(hidden_size, no_epochs, data_gen,
                                 lambda a: a, coreset_size, batch_size, single_head, val=True)
        vcl_h50_accs[i, :, :] = vcl_result_h50

    _ibp_acc = np.nanmean(vcl_ibp_accs, (0, 1))
    _vcl_result_h10 = np.nanmean(vcl_h10_accs, (0, 1))
    _vcl_result_h5 = np.nanmean(vcl_h5_accs, (0, 1))
    _vcl_result_h50 = np.nanmean(vcl_h50_accs, (0, 1))
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    plt.plot(np.arange(len(_ibp_acc)) + 1, _ibp_acc, label='VCL + IBP', marker='o')
    plt.plot(np.arange(len(_ibp_acc)) + 1, _vcl_result_h10, label='VCL h10', marker='o')
    plt.plot(np.arange(len(_ibp_acc)) + 1, _vcl_result_h5, label='VCL h5', marker='o')
    plt.plot(np.arange(len(_ibp_acc)) + 1, _vcl_result_h50, label='VCL h50', marker='o')
    ax.set_xticks(range(1, len(_ibp_acc) + 1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.set_title('Split notMnist')
    ax.legend()
    fig.savefig('split_not_mnist_accs_{}.png'.format(args.tag), bbox_inches='tight')
    plt.close()

    with open('results/split_not_mnist_res5_{}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'vcl_h10': vcl_h10_accs,
                     'vcl_h5': vcl_h5_accs,
                     'vcl_h50': vcl_h50_accs}, input_file)

    print("Finished running.")