import numpy as np
import tensorflow as tf
import pickle
import pdb
import os.path
import sys
import argparse
sys.path.extend(['alg/'])
from run_split import SplitMnistGenerator
from vcl import run_vcl, run_vcl_ibp
from utils import get_scores, concatenate_results
from visualise import plot_uncertainties, plot_Zs
from copy import deepcopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class NotMnistGenerator(SplitMnistGenerator):
    # train_size = 200000, valid_size = 10000, test_size = 10000
    def __init__(self, val, noise=False, cl3=False):

        super(SplitMnistGenerator, self).__init__(val=val, cl3=cl3)

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
    parser.add_argument('--runs', action='store',
                        dest='runs',
                        default=1,
                        type=int,
                        help='Number optmisations to perform.')
    parser.add_argument('--log_dir', action='store',
                        dest='log_dir',
                        default='logs',
                        help='TB log directory.')
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
    parser.add_argument('--h', nargs='+',
                        dest='h_list',
                        type=int,
                        default=[5, 50],
                        help='List of hidden states')
    parser.add_argument('--cl3', naction='store_true',
                        dest='cl3',
                        default=False,
                        help='Whether to use incremental class learning')
    args = parser.parse_args()

    print('tag                  = {!r}'.format(args.tag))
    print('noise                = {!r}'.format(args.noise))
    print('single_head          = {!r}'.format(args.single_head))
    print('implicit_beta        = {!r}'.format(args.implicit_beta))
    print('num_layers           = {!r}'.format(args.num_layers))
    print('use_local_reparam    = {!r}'.format(args.use_local_reparam))
    print('alpha0               = {!r}'.format(args.alpha0))
    print('hibp                 = {!r}'.format(args.hibp))
    print('log_dir              = {!r}'.format(args.log_dir))
    print('h_list               = {!r}'.format(args.h_list))
    print('run_baselines        = {!r}'.format(args.run_baselines))
    print('cl3                  = {!r}'.format(args.cl3))

    seeds = list(range(10, 10 + args.runs))
    num_tasks = 5

    vcl_ibp_accs = np.zeros((len(seeds), num_tasks, num_tasks))
    baseline_accs = {h: np.zeros((len(seeds), num_tasks, num_tasks)) for h in args.h_list}
    all_ibp_uncerts = np.zeros((len(seeds), num_tasks, num_tasks))
    baseline_uncerts = {h: np.zeros((len(seeds), num_tasks, num_tasks)) for h in args.h_list}
    all_Zs = []

    # Beta and Concrete params
    alpha0 = args.alpha0
    beta0 = 1.0
    lambda_1 = 0.5
    lambda_2 = 1.0
    # Gaussian params
    prior_mean = 0.0
    prior_var = 0.1

    for i in range(len(seeds)):
        s = seeds[i]
        hidden_size = [100]*args.num_layers
        batch_size = 256
        no_epochs = 1000
        ibp_samples = 10
        no_pred_samples = 100

        tf.compat.v1.set_random_seed(s)
        np.random.seed(1)

        coreset_size = 0
        data_gen = NotMnistGenerator(args.noise, cl3=args.cl3)
        val = True
        name = "ibp_{0}_run{1}_{2}".format("not", i + 1, args.tag)
        # Z matrix for each task is outout
        # This is overwritten for each run
        ibp_acc, Zs, uncerts = run_vcl_ibp(hidden_size=hidden_size, alphas=[1.]*len(hidden_size),
                                           no_epochs=[int(no_epochs*1.4)] + [no_epochs]*4,
                                           data_gen=data_gen, name=name, val=val, batch_size=batch_size,
                                           single_head=args.single_head, prior_mean=prior_mean, prior_var=prior_var,
                                           alpha0=alpha0, beta0=beta0, lambda_1=lambda_1, lambda_2=lambda_2,
                                           learning_rate=0.0001, no_pred_samples=no_pred_samples, ibp_samples=ibp_samples,
                                           log_dir=args.log_dir, use_local_reparam=args.use_local_reparam,
                                           implicit_beta=args.implicit_beta, hibp=args.hibp)
        all_Zs.append(Zs)
        vcl_ibp_accs[i, :, :] = ibp_acc
        all_ibp_uncerts[i, :, :] = uncerts

        if args.run_baselines:
            # Run Vanilla VCL
            for h in args.h_list:
                tf.compat.v1.reset_default_graph()
                hidden_size = [h] * args.num_layers
                data_gen = NotMnistGenerator(args.noise)
                vcl_result, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                                  lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                                  name='vcl_h{0}_not_run{1}_{2}'.format(h, i+1, args.tag),
                                                  log_dir=args.log_dir, use_local_reparam=args.use_local_reparam)
                baseline_accs[h][i, :, :] = vcl_result
                baseline_uncerts[h][i, :, :] = uncerts

    _ibp_acc = np.nanmean(vcl_ibp_accs, (0, 1))
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    plt.plot(np.arange(len(_ibp_acc)) + 1, _ibp_acc, label='VCL + IBP', marker='o')
    for h in args.h_list:
        plt.plot(np.arange(len(_ibp_acc)) + 1, np.nanmean(baseline_accs[h], (0, 1)), label='VCL h{}'.format(h), marker='o')
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
    # TODO: make plotting function cleaner
    if len(args.h_list) == 3:
        plot_uncertainties(num_tasks, all_ibp_uncerts, baseline_uncerts[args.h_list[0]],
                           baseline_uncerts[args.h_list[1]],
                           baseline_uncerts[args.h_list[2]], args.tag)

    with open('results/split_not_mnist_res5_{}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'vcl_baselines': baseline_accs,
                     'uncerts_ibp': all_ibp_uncerts,
                     'uncerts_vcl_baselines': baseline_uncerts,
                     'Z': all_Zs}, input_file)

    print("Finished running.")