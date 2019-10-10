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
            c = 1.0
        else:
            c = 0.0
        X_train = d['train_dataset'].reshape((-1, 28*28))
        self.X_train = X_train + c * np.random.uniform(0.0, 1.0, X_train.shape)
        self.train_label = d['train_labels']
        X_val = d['valid_dataset'].reshape((-1, 28*28))
        self.X_val = X_val + c * np.random.uniform(0.0, 1.0, X_val.shape)
        self.val_label = d['valid_labels']
        X_test = d['test_dataset'].reshape((-1, 28*28))
        self.X_test = X_test + c * np.random.uniform(0.0, 1.0, X_test.shape)
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
    parser.add_argument('--noise', action='store_true',
                        default=False,
                        dest='noise',
                        help='Whether use random uniform noise.')
    args = parser.parse_args()

    print('tag          = {!r}'.format(args.tag))
    print('noise        = {!r}'.format(args.noise))

    seeds = [12, 13, 14, 15, 16]

    vcl_ibp_accs = np.zeros((len(seeds), 5, 5))
    vcl_h5_accs = np.zeros((len(seeds), 5, 5))
    vcl_h10_accs = np.zeros((len(seeds), 5, 5))
    vcl_h50_accs = np.zeros((len(seeds), 5, 5))
    all_Zs = []

    # bayes_opt params
    alpha0 = 5.0
    beta0 = 1.0
    lambda_1 = 1.0
    lambda_2 = 1.0

    for i in range(len(seeds)):
        s = seeds[i]
        hidden_size = [100]
        batch_size = 256
        no_epochs = 1000
        ibp_samples = 10

        tf.set_random_seed(s)
        np.random.seed(1)

        ibp_acc = np.array([])

        coreset_size = 0
        data_gen = NotMnistGenerator(args.noise)
        single_head = False
        val = True

        # Z matrix for each task is outout
        # This is overwritten for each run
        acc, Zs = run_vcl_ibp(hidden_size=hidden_size, no_epochs=no_epochs, data_gen=data_gen,
                              run_index=i, tag=args.tag, dataset=args.dataset,
                              val=val, batch_size=None, single_head=True, alpha0=5.0,
                              beta0=1.0, lambda_1=1.0, lambda_2=1.0, learning_rate=0.0001,
                              no_pred_samples=100, ibp_samples=10)
        ibp_acc = concatenate_results(acc, ibp_acc)
        all_Zs.append(Zs)
        vcl_ibp_accs[i, :, :] = ibp_acc

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [10]
        data_gen = NotMnistGenerator(args.noise)
        vcl_result_h10 = run_vcl(hidden_size, no_epochs, data_gen,
                                 lambda a: a, coreset_size, batch_size, single_head, val=val,
                                 name='vcl_h10_{0}_run{1}'.format(args.tag, i + 1))
        vcl_h10_accs[i, :, :] = vcl_result_h10

        tf.reset_default_graph()
        hidden_size = [5]
        data_gen = NotMnistGenerator(args.noise)
        vcl_result_h5 = run_vcl(hidden_size, no_epochs, data_gen,
                                lambda a: a, coreset_size, batch_size, single_head, val=val,
                                name='vcl_h5_{0}_run{1}'.format(args.tag, i + 1))
        vcl_h5_accs[i, :, :] = vcl_result_h5

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [50]
        data_gen = NotMnistGenerator(args.noise)
        vcl_result_h50 = run_vcl(hidden_size, no_epochs, data_gen,
                                 lambda a: a, coreset_size, batch_size, single_head, val=val,
                                 name='vcl_h50_{0}_run{1}'.format(args.tag, i + 1))
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

    # this is plotting the final run Zs
    num_tasks = 5
    fig, ax = plt.subplots(2, num_tasks, figsize=(16, 4))
    for i in range(num_tasks):
        ax[0][i].imshow(np.squeeze(Zs[i])[:50, :], cmap=plt.cm.Greys_r, vmin=0, vmax=1)
        ax[0][i].set_xticklabels([])
        ax[0][i].set_yticklabels([])
        ax[1][i].hist(np.sum(np.squeeze(Zs[i]), axis=1), 10)
        ax[1][i].set_yticklabels([])
        ax[1][i].set_xlabel("Task {}".format(i + 1))
        plt.savefig('plots/Zs_{0}.pdf'.format(args.tag), bbox_inches='tight')
        fig.show()

    print("Prop of neurons which are active for each task: ", [np.mean(Zs[i]) for i in range(num_tasks)])

    with open('results/split_not_mnist_res5_{}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'vcl_h10': vcl_h10_accs,
                     'vcl_h5': vcl_h5_accs,
                     'vcl_h50': vcl_h50_accs,
                     'Z': all_Zs}, input_file)

    print("Finished running.")