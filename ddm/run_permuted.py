import numpy as np
import tensorflow as tf
import gzip
import sys
import argparse
sys.path.extend(['alg/'])
from vcl import run_vcl, run_vcl_ibp
from utils import get_scores, concatenate_results
from visualise import plot_uncertainties, plot_Zs
import pickle
from copy import deepcopy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class PermutedMnistGenerator():
    def __init__(self, max_iter=5, val=False):

        self.val = val
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        if self.val:
            self.X_train = train_set[0]
            self.X_val = valid_set[0]
            self.Y_train = train_set[1]
            self.Y_val = valid_set[1]
        else:
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
            next_x_test = next_x_test[:, perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            if self.val:
                next_x_val = deepcopy(self.X_val)
                next_x_val = next_x_val[:, perm_inds]
                next_y_val = np.eye(10)[self.Y_val]

                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val
            else:
                return next_x_train, next_y_train, next_x_test, next_y_test

    def reset_cur_iter(self):
        self.cur_iter = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_head', action='store_true',
                        default=False,
                        dest='single_head',
                        help='Whether to use a single head.')
    parser.add_argument('--implicit_beta', action='store_true',
                        default=False,
                        dest='implicit_beta',
                        help='Whether to use reparam for Beta dist.')
    parser.add_argument('--hibp', action='store_true',
                        default=False,
                        dest='hibp',
                        help='Whether to use hibp.')
    parser.add_argument('--num_layers', action='store',
                        dest='num_layers',
                        default=1,
                        type=int,
                        help='Number of layers in the NNs.')
    parser.add_argument('--log_dir', action='store',
                        dest='log_dir',
                        default='logs',
                        help='TB Log directory.')
    parser.add_argument('--run_baselines', action='store_true',
                        default=False,
                        dest='run_baselines',
                        help='Whether to run VCL baselines.')
    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')
    parser.add_argument('--h', nargs='+',
                        dest='h_list',
                        type=int,
                        default=[5, 50],
                        help='List of hidden states')
    parser.add_argument('--K', action='store',
                        dest='K',
                        type=int,
                        default=100,
                        help='Variational truncation param for IBP.')
    args = parser.parse_args()

    print('single_head          = {!r}'.format(args.single_head))
    print('implicit_beta        = {!r}'.format(args.implicit_beta))
    print('num_layers           = {!r}'.format(args.num_layers))
    print('hibp                 = {!r}'.format(args.hibp))
    print('log_dir              = {!r}'.format(args.log_dir))
    print('tag                  = {!r}'.format(args.tag))
    print('run_baselines        = {!r}'.format(args.run_baselines))
    print('h_list               = {!r}'.format(args.h_list))
    print('K                    = {!r}'.format(args.K))

    seeds = [12, 13, 14, 15, 16]
    num_tasks = 5

    vcl_ibp_accs = np.zeros((len(seeds), num_tasks, num_tasks))
    baseline_accs = {h: np.zeros((len(seeds), num_tasks, num_tasks)) for h in args.h_list}
    all_ibp_uncerts = np.zeros((len(seeds), num_tasks, num_tasks))
    baseline_uncerts = {h: np.zeros((len(seeds), num_tasks, num_tasks)) for h in args.h_list}
    all_Zs = []

    alpha0 = 5.0
    beta0 = 1.0
    lambda_1 = 1.0
    lambda_2 = 1.0
    alpha = 4.0

    hidden_size = [args.K] * args.num_layers
    batch_size = 512
    no_epochs = 200
    ibp_samples = 10

    val = True
    for i in range(len(seeds)):
        s = seeds[i]
        tf.set_random_seed(s)
        np.random.seed(1)

        coreset_size = 0
        data_gen = PermutedMnistGenerator(num_tasks, val=val)
        name = "ibp_{0}_run{1}_{2}".format("perm", i + 1, args.tag)
        # Z matrix for each task is output
        # This is overwritten for each run
        ibp_acc, Zs, uncerts = run_vcl_ibp(hidden_size=hidden_size, alphas=[alpha]*len(hidden_size),
                                           no_epochs=[no_epochs]*num_tasks,
                                           data_gen=data_gen, name=name, val=val, batch_size=batch_size,
                                           single_head=args.single_head, alpha0=alpha0, beta0=beta0,
                                           lambda_1=lambda_1, lambda_2=lambda_2,
                                           learning_rate=0.001, no_pred_samples=100, ibp_samples=ibp_samples,
                                           log_dir=args.log_dir,
                                           implicit_beta=args.implicit_beta, hibp=args.hibp)
        all_Zs.append(Zs)
        vcl_ibp_accs[i, :, :] = ibp_acc
        all_ibp_uncerts[i, :, :] = uncerts

        # Run Vanilla VCL
        if args.run_baselines:
            for h in args.h_list:
                tf.reset_default_graph()
                hidden_size = [h] * args.num_layers
                data_gen = PermutedMnistGenerator(num_tasks, val=val)
                vcl_result, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                              lambda a: a, coreset_size, batch_size, args.single_head, val=val,
                                               name='vcl_perm_h{0}_{1}_run{2}'.format(h, args.tag, i + 1),
                                               log_dir=args.log_dir)
                baseline_accs[h][i, :, :] = vcl_result
                baseline_uncerts[h][i, :, :] = uncerts

    _ibp_acc = np.nanmean(vcl_ibp_accs, (0, 1))
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    plt.plot(np.arange(len(_ibp_acc)) + 1, _ibp_acc, label='VCL + IBP', marker='o')
    for h in args.h_list:
        plt.plot(np.arange(len(_ibp_acc)) + 1, np.nanmean(baseline_accs[h], (0, 1)), label='VCL h{}'.format(h),
                 marker='o')
    ax.set_xticks(range(1, len(_ibp_acc) + 1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.set_title('Permuted MNIST')
    ax.legend()
    fig.savefig('permuted_mnist_accs_{}.png'.format(args.tag), bbox_inches='tight')
    plt.close()

    print("Prop of neurons which are active for each task (and layer):",
          [np.mean(Zs[i]) for i in range(num_tasks * args.num_layers)])

    # Uncertainties
    # TODO: make plotting function cleaner
    if len(args.h_list) == 3:
        plot_uncertainties(num_tasks, all_ibp_uncerts, baseline_uncerts[args.h_list[0]],
                           baseline_uncerts[args.h_list[1]],
                           baseline_uncerts[args.h_list[2]], args.tag)

    with open('results/permuted_mnist_res5_{}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'vcl_baselines': baseline_accs,
                     'uncerts_ibp': all_ibp_uncerts,
                     'uncerts_vcl_baselines': baseline_uncerts,
                     'Z': all_Zs}, input_file)

    print("Finished running.")
