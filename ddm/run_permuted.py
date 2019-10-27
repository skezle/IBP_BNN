import numpy as np
import tensorflow as tf
import gzip
import sys
import argparse
sys.path.extend(['alg/'])
from vcl import run_vcl, run_vcl_ibp
from cla_models_multihead import Vanilla_NN, MFVI_IBP_NN
from utils import get_scores, concatenate_results
from visualise import plot_uncertainties
import pickle
from copy import deepcopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')
    args = parser.parse_args()

    print('tag          = {!r}'.format(args.tag))

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

    alpha0 = 5.0
    beta0 = 1.0
    lambda_1 = 1.0
    lambda_2 = 1.0

    num_tasks = 5

    for i in range(len(seeds)):
        s = seeds[i]
        hidden_size = [100]
        batch_size = 256
        no_epochs = 1000
        ibp_samples = 10

        tf.set_random_seed(s)
        np.random.seed(1)

        val = False
        coreset_size = 0
        data_gen = PermutedMnistGenerator(num_tasks)
        single_head = True
        name = "ibp_{0}_run{1}_{2}".format("perm", i + 1, args.tag)
        # Z matrix for each task is output
        # This is overwritten for each run
        ibp_acc, Zs, uncerts = run_vcl_ibp(hidden_size=hidden_size, no_epochs=no_epochs, data_gen=data_gen,
                                           name=name,
                                           val=val, batch_size=None, single_head=True, alpha0=alpha0,
                                           beta0=beta0, lambda_1=lambda_1, lambda_2=lambda_2, learning_rate=0.0001,
                                           no_pred_samples=100, ibp_samples = 10)
        all_Zs.append(Zs)
        vcl_ibp_accs[i, :, :] = ibp_acc
        all_ibp_uncerts[i, :, :] = uncerts

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [10, 10]
        data_gen = PermutedMnistGenerator(num_tasks)
        vcl_result_h10, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                          lambda a: a, coreset_size, batch_size, single_head, val=False,
                                          name='vcl_perm_h10_{0}_run{1}'.format(args.tag, i + 1))
        vcl_h10_accs[i, :, :] = vcl_result_h10
        all_vcl_h10_uncerts[i, :, :] = uncerts

        tf.reset_default_graph()
        hidden_size = [5, 5]
        data_gen = PermutedMnistGenerator(num_tasks)
        vcl_result_h5, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                lambda a: a, coreset_size, batch_size, single_head, val=False,
                                name='vcl_perm_h5_{0}_run{1}'.format(args.tag, i + 1))
        vcl_h5_accs[i, :, :] = vcl_result_h5
        all_vcl_h5_uncerts[i, :, :] = uncerts

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [50, 50]
        data_gen = PermutedMnistGenerator(num_tasks)
        vcl_result_h50, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                 lambda a: a, coreset_size, batch_size, single_head, val=False,
                                 name='vcl_perm_h50_{0}_run{1}'.format(args.tag, i + 1))
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
    ax.set_title('Permuted MNIST')
    ax.legend()
    fig.savefig('permuted_mnist_accs_{}.png'.format(args.tag), bbox_inches='tight')
    plt.close()

    num_tasks = 5
    fig, ax = plt.subplots(2, num_tasks, figsize=(16, 4))
    for i in range(num_tasks):
        ax[0][i].imshow(np.squeeze(Zs[i])[:50, :], cmap=plt.cm.Greys_r, vmin=0, vmax=1)
        ax[0][i].set_xticklabels([])
        ax[0][i].set_yticklabels([])
        ax[1][i].hist(np.sum(np.squeeze(Zs[i]), axis=1), 10)
        ax[1][i].set_yticklabels([])
        ax[1][i].set_xlabel("Task {}".format(i + 1))
        plt.savefig('plots/Zs_perm_{0}.pdf'.format(args.tag), bbox_inches='tight')
        fig.show()

    print("Prop of neurons which are active for each task: ", [np.mean(Zs[i]) for i in range(num_tasks)])

    # Uncertainties
    plot_uncertainties(num_tasks, all_ibp_uncerts, all_vcl_h5_uncerts, all_vcl_h10_uncerts,
                       all_vcl_h50_uncerts, args.tag)

    with open('results/permuted_mnist_res5_{}.pkl'.format(args.tag), 'wb') as input_file:
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
