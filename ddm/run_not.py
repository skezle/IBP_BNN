import numpy as np
import tensorflow as tf
import pickle
import pdb
import os.path
import sys
import argparse
sys.path.extend(['alg/'])
from cla_models_multihead import Vanilla_NN, MFVI_IBP_NN
from vcl import run_vcl, run_vcl_ibp
from utils import get_scores, concatenate_results
from visualise import plot_uncertainties, plot_Zs
from copy import deepcopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class NotMnistGenerator:
    # train_size = 200000, valid_size = 10000, test_size = 10000
    def __init__(self, noise=False):
        self.noise = noise
        with open('data/notMNIST.pickle', 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        if self.noise:
            # train
            X_train = d['train_dataset'].reshape((-1, 28 * 28))
            idx = np.greater(np.ones(X_train.shape) * 0.5,
                             X_train)  # False where the character lies and True in the background
            new = np.zeros(X_train.shape)
            new[idx] = 1.0
            self.X_train = X_train + new * np.random.uniform(np.min(X_train), np.max(X_train),
                                                                  new.shape)
            self.train_label = d['train_labels']
            # validation
            X_val = d['valid_dataset'].reshape((-1, 28 * 28))
            idx = np.greater(np.ones(X_val.shape) * 0.5,
                             X_val)  # False where the character lies and True in the background
            new = np.zeros(X_val.shape)
            new[idx] = 1.0
            self.X_val = X_val + new * np.random.uniform(np.min(X_val), np.max(X_val),
                                                         X_val.shape)
            self.val_label = d['valid_labels']
            # test
            X_test = d['test_dataset'].reshape((-1, 28 * 28))
            idx = np.greater(np.ones(X_test.shape) * 0.5,
                             X_test)  # False where the character lies and True in the background
            new = np.zeros(X_test.shape)
            new[idx] = 1.0
            self.X_test = X_test + new * np.random.uniform(np.min(X_test), np.max(X_test),
                                                           X_test.shape)
            self.test_label = d['test_labels']
        else:
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

    def reset_cur_iter(self):
        self.cur_iter = 0

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
    parser.add_argument('--noise', action='store_true',
                        default=False,
                        dest='noise',
                        help='Whether use random uniform noise.')
    parser.add_argument('--single_head', action='store_true',
                        default=False,
                        dest='single_head',
                        help='Whether to use a single head.')
    parser.add_argument('--num_layers', action='store',
                        dest='num_layers',
                        default=1,
                        type=int,
                        help='Number of layers in the NNs.')
    parser.add_argument('--log_dir', action='store',
                        dest='log_dir',
                        default='logs',
                        help='TB log directory.')
    args = parser.parse_args()

    print('tag          = {!r}'.format(args.tag))
    print('noise        = {!r}'.format(args.noise))
    print('single_head  = {!r}'.format(args.single_head))
    print('num_layers   = {!r}'.format(args.num_layers))
    print('log_dir      = {!r}'.format(args.log_dir))

    seeds = [12, 13, 14, 15, 16]
    num_tasks = 5

    vcl_ibp_accs = np.zeros((len(seeds), num_tasks, num_tasks))
    vcl_h5_accs = np.zeros((len(seeds), num_tasks, num_tasks))
    vcl_h10_accs = np.zeros((len(seeds), num_tasks, num_tasks))
    vcl_h50_accs = np.zeros((len(seeds), num_tasks, num_tasks))
    all_ibp_uncerts = np.zeros((len(seeds), num_tasks, num_tasks))
    all_vcl_h5_uncerts = np.zeros((len(seeds), num_tasks, num_tasks))
    all_vcl_h10_uncerts = np.zeros((len(seeds), num_tasks, num_tasks))
    all_vcl_h50_uncerts = np.zeros((len(seeds), num_tasks, num_tasks))
    all_Zs = []

    # bayes_opt params
    alpha0 = 5.0
    beta0 = 1.0
    lambda_1 = 1.0
    lambda_2 = 1.0

    for i in range(len(seeds)):
        s = seeds[i]
        hidden_size = [100]*args.num_layers
        batch_size = 256
        no_epochs = 1000
        ibp_samples = 10
        no_pred_samples = 100

        tf.set_random_seed(s)
        np.random.seed(1)

        coreset_size = 0
        data_gen = NotMnistGenerator(args.noise)
        val = True
        name = "ibp_{0}_run{1}_{2}".format("not", i + 1, args.tag)
        # Z matrix for each task is outout
        # This is overwritten for each run
        ibp_acc, Zs, uncerts = run_vcl_ibp(hidden_size=hidden_size, no_epochs=[no_epochs*2] + [no_epochs]*4,
                                           data_gen=data_gen, name=name, val=val, batch_size=batch_size,
                                           single_head=args.single_head,
                                           alpha0=alpha0, beta0=beta0, lambda_1=lambda_1, lambda_2=lambda_2,
                                           learning_rate=0.0001, no_pred_samples=no_pred_samples, ibp_samples=ibp_samples,
                                           log_dir=args.log_dir)
        all_Zs.append(Zs)
        vcl_ibp_accs[i, :, :] = ibp_acc
        all_ibp_uncerts[i, :, :] = uncerts

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [10]*args.num_layers
        data_gen = NotMnistGenerator(args.noise)
        vcl_result_h10, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                          lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                          name='vcl_h10_{0}_run{1}'.format(args.tag, i + 1),
                                          log_dir=args.log_dir)
        vcl_h10_accs[i, :, :] = vcl_result_h10
        all_vcl_h10_uncerts[i, :, :] = uncerts

        tf.reset_default_graph()
        hidden_size = [5]*args.num_layers
        data_gen = NotMnistGenerator(args.noise)
        vcl_result_h5, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                         lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                         name='vcl_h5_{0}_run{1}'.format(args.tag, i + 1),
                                         log_dir=args.log_dir)
        vcl_h5_accs[i, :, :] = vcl_result_h5
        all_vcl_h5_uncerts[i, :, :] = uncerts

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [50]*args.num_layers
        data_gen = NotMnistGenerator(args.noise)
        vcl_result_h50, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                          lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                          name='vcl_h50_{0}_run{1}'.format(args.tag, i + 1),
                                          log_dir=args.log_dir)
        vcl_h50_accs[i, :, :] = vcl_result_h50
        all_vcl_h50_uncerts[i, :, :] = uncerts

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

    # were are only plotting the final optimisation of Zs (remember there are 5 repetitions of this)
    print("length of Zs: {}".format(len(Zs)))
    plot_Zs(num_tasks, args.num_layers, Zs, "not", args.tag)
    print("Prop of neurons which are active for each task (and layer):", [np.mean(Zs[i]) for i in range(num_tasks*args.num_layers)])

    # Uncertainties
    plot_uncertainties(num_tasks, all_ibp_uncerts, all_vcl_h5_uncerts, all_vcl_h10_uncerts,
                       all_vcl_h50_uncerts, args.tag)

    with open('results/split_not_mnist_res5_{}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'vcl_h10': vcl_h10_accs,
                     'vcl_h5': vcl_h5_accs,
                     'vcl_h50': vcl_h50_accs,
                     'uncerts_ibp': all_ibp_uncerts,
                     'uncerts_vcl_h5': all_vcl_h5_uncerts,
                     'uncerts_vcl_h10': all_vcl_h10_uncerts,
                     'uncerts_vcl_h50': all_vcl_h50_uncerts,
                     'Z': all_Zs}, input_file)

    print("Finished running.")