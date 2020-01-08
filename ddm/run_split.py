import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
import os.path
import argparse
sys.path.extend(['alg/'])
from vcl import run_vcl, run_vcl_ibp
from utils import concatenate_results, get_scores
from visualise import plot_uncertainties, plot_Zs
from copy import deepcopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SplitMnistBackgroundGenerator:
    """ Thanks https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits
    """
    def __init__(self, val=False):
        self.val = val
        # 12000 train, 50000 test
        train = np.loadtxt('data/mnist_background_images/mnist_background_images_train.amat')
        test = np.loadtxt('data/mnist_background_images/mnist_background_images_test.amat')
        all = np.vstack((train, test))
        if self.val:
            n_train = 40000
            n_test = 10000
            n_val = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:(n_train+n_test), :-1]
            self.test_label = all[n_train:(n_train+n_test), -1:]
            self.X_val = all[(n_train+n_test):, :-1]
            self.val_label = all[(n_train+n_test):, -1:]
        else:
            n_train = 52000
            n_test = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:, :-1]
            self.test_label = all[n_train:, -1:]

        assert self.X_train.shape[1] == 28 * 28
        assert self.X_test.shape[1] == 28 * 28

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
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

            if self.val:
                val_0_id = np.where(self.val_label == self.sets_0[self.cur_iter])[0]
                val_1_id = np.where(self.val_label == self.sets_1[self.cur_iter])[0]
                next_x_val = np.vstack((self.X_val[val_0_id], self.X_val[val_1_id]))

                next_y_val = np.vstack((np.ones((val_0_id.shape[0], 1)), np.zeros((val_1_id.shape[0], 1))))
                next_y_val = np.hstack((next_y_val, 1 - next_y_val))
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val
            else:
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test

    def reset_cur_iter(self):
        self.cur_iter = 0


class SplitMnistRandomGenerator:
    """ Thanks https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits

    """
    def __init__(self, val=False):
        self.val = val
        # 12000 train, 50000 test
        train = np.loadtxt('data/mnist_background_random/mnist_background_random_train.amat')
        test = np.loadtxt('data/mnist_background_random/mnist_background_random_test.amat')
        all = np.vstack((train, test))
        if self.val:
            n_train = 40000
            n_test = 10000
            n_val = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:(n_train+n_test), :-1]
            self.test_label = all[n_train:(n_train+n_test), -1:]
            self.X_val = all[(n_train+n_test):, :-1]
            self.val_label = all[(n_train+n_test):, -1:]
        else:
            n_train = 52000
            n_test = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:, :-1]
            self.test_label = all[n_train:, -1:]

        assert self.X_train.shape[1] == 28 * 28
        assert self.X_test.shape[1] == 28 * 28

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
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

            if self.val:
                val_0_id = np.where(self.val_label == self.sets_0[self.cur_iter])[0]
                val_1_id = np.where(self.val_label == self.sets_1[self.cur_iter])[0]
                next_x_val = np.vstack((self.X_val[val_0_id], self.X_val[val_1_id]))

                next_y_val = np.vstack((np.ones((val_0_id.shape[0], 1)), np.zeros((val_1_id.shape[0], 1))))
                next_y_val = np.hstack((next_y_val, 1 - next_y_val))
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val
            else:
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test

    def reset_cur_iter(self):
        self.cur_iter = 0


class SplitMnistGenerator:
    def __init__(self, val=False, num_tasks=5, difficult=False):
        # train, val, test (50000, 784) (10000, 784) (10000, 784)
        self.val = val
        self.num_tasks = num_tasks
        self.difficult = difficult # make the hardest task the first to see if the number of active neurons can shrink
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        if self.val:
            self.X_train = train_set[0]
            self.X_val = valid_set[0]
            self.train_label = train_set[1]
            self.val_label = valid_set[1]
        else:
            self.X_train = np.vstack((train_set[0], valid_set[0]))
            self.train_label = np.hstack((train_set[1], valid_set[1]))

        self.X_test = test_set[0]
        self.test_label = test_set[1]

        if self.num_tasks == 1:
            self.sets_0 = [0]
            self.sets_1 = [1]
        else:
            if self.difficult:
                self.sets_0 = [4, 0, 2, 6, 8]
                self.sets_1 = [5, 1, 3, 7, 9]
            else:
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

            if self.val:
                val_0_id = np.where(self.val_label == self.sets_0[self.cur_iter])[0]
                val_1_id = np.where(self.val_label == self.sets_1[self.cur_iter])[0]
                next_x_val = np.vstack((self.X_val[val_0_id], self.X_val[val_1_id]))

                next_y_val = np.vstack((np.ones((val_0_id.shape[0], 1)), np.zeros((val_1_id.shape[0], 1))))
                next_y_val = np.hstack((next_y_val, 1 - next_y_val))
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val
            else:
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test

    def reset_cur_iter(self):
        self.cur_iter = 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--difficult', action='store_true',
                        default=False,
                        dest='difficult',
                        help='Whether to start with the most difficult task.')
    parser.add_argument('--single_head', action='store_true',
                        default=False,
                        dest='single_head',
                        help='Whether to use a single head.')
    parser.add_argument('--num_layers', action='store',
                        dest='num_layers',
                        default=1,
                        type=int,
                        help='Number of layers in the NNs.')
    parser.add_argument('--runs', action='store',
                        dest='runs',
                        default=1,
                        type=int,
                        help='Number optmisations to perform.')
    parser.add_argument('--log_dir', action='store',
                        dest='log_dir',
                        default='logs',
                        help='TB log directory.')
    parser.add_argument('--dataset', action='store',
                        dest='dataset',
                        help='Which dataset to choose {normal, noise, background}.')
    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')
    parser.add_argument('--use_local_reparam', action='store_true',
                        default=False,
                        dest='use_local_reparam',
                        help='Whether to use local reparam.')
    parser.add_argument('--alpha0', action='store',
                        default=5.0,
                        type=float,
                        dest='alpha0',
                        help='The prior and initialisation for the beta concentration param.')
    parser.add_argument('--implicit_beta', action='store_true',
                        default=False,
                        dest='implicit_beta',
                        help='Whether to use reparam for Beta dist.')
    parser.add_argument('--hibp', action='store_true',
                        default=False,
                        dest='hibp',
                        help='Whether to use hibp.')
    parser.add_argument('--run_baselines', action='store_true',
                        default=False,
                        dest='run_baselines',
                        help='Whether to run the baselines.')
    args = parser.parse_args()

    print('difficult            = {!r}'.format(args.difficult))
    print('single_head          = {!r}'.format(args.single_head))
    print('num_layers           = {!r}'.format(args.num_layers))
    print('runs                 = {!r}'.format(args.runs))
    print('alpha0               = {!r}'.format(args.alpha0))
    print('log_dir              = {!r}'.format(args.log_dir))
    print('dataset              = {!r}'.format(args.dataset))
    print('use_local_reparam    = {!r}'.format(args.use_local_reparam))
    print('implicit_beta        = {!r}'.format(args.implicit_beta))
    print('hibp                 = {!r}'.format(args.hibp))
    print('run_baselines        = {!r}'.format(args.run_baselines))
    print('tag                  = {!r}'.format(args.tag))

    seeds = list(range(1, 1 + args.runs))
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

    # We don't need a validation set
    val = False

    def get_datagen():
        if args.dataset == 'normal':
            data_gen = SplitMnistGenerator(val=val, difficult=args.difficult)
        elif args.dataset == 'random':
            data_gen = SplitMnistRandomGenerator(val=val)
        elif args.dataset == 'background':
            data_gen = SplitMnistBackgroundGenerator(val=val)
        else:
            raise ValueError('Pick dataset in {normal, random, background}')
        return data_gen

    # IBP params
    alpha0 = 4.2
    beta0 = 1.0
    lambda_1 = 0.5
    lambda_2 = 0.7
    alpha = 4.0
    # Gaussian params
    prior_mean = 0.0
    prior_var = 0.7

    for i in range(len(seeds)):
        s = seeds[i]
        hidden_size = [100] * args.num_layers
        batch_size = 512
        no_epochs = 600
        ibp_samples = 10
        no_pred_samples = 100

        tf.set_random_seed(s)
        np.random.seed(1)

        coreset_size = 0
        data_gen = get_datagen()
        name = "split_{0}_run{1}_{2}".format(args.dataset, i + 1, args.tag)
        # Z matrix for each task is output
        # This is overwritten for each run
        ibp_acc, Zs, uncerts = run_vcl_ibp(hidden_size=hidden_size, alphas=[1.]*len(hidden_size),
                                           no_epochs= [no_epochs*1.5] + [no_epochs]*(num_tasks-1), data_gen=data_gen,
                                           name=name, val=val, batch_size=batch_size, single_head=args.single_head,
                                           prior_mean=prior_mean, prior_var=prior_var, alpha0=alpha0,
                                           beta0=beta0, lambda_1=lambda_1, lambda_2=lambda_2,
                                           learning_rate=[0.0001]*num_tasks,
                                           no_pred_samples=no_pred_samples, ibp_samples=ibp_samples, log_dir=args.log_dir,
                                           use_local_reparam=args.use_local_reparam,
                                           implicit_beta=args.implicit_beta, hibp=args.hibp)

        all_Zs.append(Zs)
        vcl_ibp_accs[i, :, :] = ibp_acc
        all_ibp_uncerts[i, :, :] = uncerts

        if args.run_baselines:
            # Run Vanilla VCL
            # Comparison with other single layer neural networks
            tf.reset_default_graph()
            hidden_size = [10] * args.num_layers
            data_gen = get_datagen()
            vcl_result_h10, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                              lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                              name='vcl_h10_{2}_run{0}_{1}'.format(i+1, args.tag, args.dataset),
                                              log_dir=args.log_dir, use_local_reparam=args.use_local_reparam)
            vcl_h10_accs[i, :, :] = vcl_result_h10
            all_vcl_h10_uncerts[i, :, :] = uncerts

            tf.reset_default_graph()
            hidden_size = [5] * args.num_layers
            data_gen = get_datagen()
            vcl_result_h5, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                             lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                             name='vcl_h5_{2}_run{0}_{1}'.format(i+1, args.tag, args.dataset),
                                             log_dir=args.log_dir, use_local_reparam=args.use_local_reparam)
            vcl_h5_accs[i, :, :] = vcl_result_h5
            all_vcl_h5_uncerts[i, :, :] = uncerts

            tf.reset_default_graph()
            hidden_size = [50] * args.num_layers
            data_gen = get_datagen()
            vcl_result_h50, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                              lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                              name='vcl_h50_{2}_run{0}_{1}'.format(i + 1, args.tag, args.dataset),
                                              log_dir=args.log_dir, use_local_reparam=args.use_local_reparam)
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
    ax.set_title('Split Mnist')
    ax.legend()
    fig.savefig('plots/split_mnist_accs_{}.png'.format(args.tag), bbox_inches='tight')
    plt.close()

    # we are only plotting the results from the final optimisation
    print("length of Zs: {}".format(len(Zs)))
    plot_Zs(num_tasks, args.num_layers, Zs, args.dataset, args.tag)
    print("Prop of neurons which are active for each task (and layer):", [np.mean(Zs[i]) for i in range(num_tasks*args.num_layers)])

    # Uncertainties
    plot_uncertainties(num_tasks, all_ibp_uncerts, all_vcl_h5_uncerts, all_vcl_h10_uncerts, all_vcl_h50_uncerts, args.tag)

    with open('results/split_mnist_res5_{}.pkl'.format(args.tag), 'wb') as input_file:
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
