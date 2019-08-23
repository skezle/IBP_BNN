import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
import copy
import os.path
import argparse
from ddm.run_split import SplitMnistGenerator
from ddm.run_not import NotMnistGenerator
from ddm.alg.cla_models_multihead import MFVI_IBP_NN, Vanilla_NN
from ddm.alg.utils import get_scores, concatenate_results
from ddm.alg.vcl import run_vcl
from copy import deepcopy

from bayes_opt import BayesianOptimization

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_data_generator():
    if args.dataset == 'split':
        val = True
        data_gen = SplitMnistGenerator(val)
    elif args.dataset == 'not':
        data_gen = NotMnistGenerator()
    else:
        raise ValueError('No dataset: {}'.format(args.dataset))

    return data_gen


def folder_name(experiment_name, param_bounds, bo_params, model_params, results_folder="./results"):
    pp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(param_bounds.items()))[:-1]
    bp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(bo_params.items()))[:-1]
    mp = ''.join('{}:{}|'.format(key, val) for key, val in sorted(model_params.items()))[:-1]
    return os.path.join(results_folder, experiment_name, pp, bp, mp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--beta_1', action='store',
                        dest='beta_1',
                        type=float,
                        default=1.0,
                        help='Gauss KL coefficient.')
    parser.add_argument('--beta_2', action='store',
                        dest='beta_2',
                        type=float,
                        default=1.0,
                        help='Beta KL coefficient.')
    parser.add_argument('--beta_3', action='store',
                        dest='beta_3',
                        type=float,
                        default=1.0,
                        help='Bernoulli KL coefficient.')
    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')

    parser.add_argument('--dataset', action='store',
                        dest='dataset',
                        default='split',
                        help='String to desribe the dataset to use for LL.')

    args = parser.parse_args()

    print('beta_1       = {!r}'.format(args.beta_1))
    print('beta_2       = {!r}'.format(args.beta_2))
    print('beta_3       = {!r}'.format(args.beta_3))
    print('dataset      = {!r}'.format(args.dataset))
    print('tag          = {!r}'.format(args.tag))

    experiment_name = 'ibp_split_mnist_bo_{}'.format(args.tag)

    bo_params = {'acq': 'ei',
                 'init_points': 5,
                 'n_iter': 10}

    param_bounds = {'alpha': (1.0, 5.),
                    'beta': (0.1, 1.),
                    'lambda_1': (0.1, 1.0),
                    'lambda_2': (0.1, 1.0)}

    model_params = {'hidden_size': [100],
                    'batch_size': 128,
                    'no_epochs': 500,
                    'learning_rate': 0.0001,
                    'anneal_rate': 0.0,
                    'pred_samples': 100,
                    'num_ibp_samples': 10,
                    'beta_1': args.beta_1,
                    'beta_2': args.beta_2,
                    'beta_3': args.beta_3}

    ############
    ## Run BO ##
    ############
    def cv_exp(alpha, beta, lambda_1, lambda_2):
        """ Runs BayesOpt on Split MNIST for lifelong learning with the BNN+IBP prior

        :params: optim params
        returns av accuracy over val set
        """
        # Run vanilla VCL
        tf.set_random_seed(12)
        np.random.seed(1)

        ibp_acc = np.array([])

        model_params_cv = copy.deepcopy(model_params)

        data_gen = get_data_generator()
        single_head = False
        in_dim, out_dim = data_gen.get_dims()
        x_valsets, y_valsets = [], []
        for task_id in range(data_gen.max_iter):

            tf.reset_default_graph()
            x_train, y_train, _, _, x_val, y_val = data_gen.next_task()
            x_valsets.append(x_val)
            y_valsets.append(y_val)

            # Set the readout head to train
            head = 0 if single_head else task_id
            bsize = x_train.shape[0] if (model_params['batch_size'] is None) else model_params['batch_size']

            # Train network with maximum likelihood to initialize first model
            if task_id == 0:
                ml_model = Vanilla_NN(in_dim, model_params_cv['hidden_size'], out_dim, x_train.shape[0])
                ml_model.train(x_train, y_train, task_id, 100,
                               model_params_cv['batch_size'], verbose=False)
                mf_weights = ml_model.get_weights()
                mf_variances = None
                mf_betas = None
                ml_model.close_session()

            # Train on non-coreset data
            mf_model = MFVI_IBP_NN(in_dim,
                                   model_params_cv['hidden_size'],
                                   out_dim,
                                   x_train.shape[0],
                                   num_ibp_samples=model_params_cv['num_ibp_samples'],
                                   prev_means=mf_weights,
                                   prev_log_variances=mf_variances,
                                   prev_betas=mf_betas,
                                   alpha0=alpha,
                                   beta0=beta,
                                   learning_rate=model_params_cv['learning_rate'],
                                   lambda_1=lambda_1,  # initial temperature of the variational posterior for task 1
                                   lambda_2=lambda_2,  # temperature of the Concrete prior
                                   no_pred_samples=model_params_cv['pred_samples'],
                                   name='{}_alpha_{:.02}_beta_{:.02}_lambda_1_{:.02}_lambda_2_{:.02}'.format(experiment_name,
                                                                                                             alpha, beta, lambda_1,
                                                                                                             lambda_2),
                                   output_tb_gradients=True,
                                   beta_1=model_params_cv['beta_1'],
                                   beta_2=model_params_cv['beta_2'],
                                   beta_3=model_params_cv['beta_3'],
                                   epsilon=0.1)

            mf_model.train(x_train, y_train, head, model_params_cv['no_epochs'],
                           model_params_cv['batch_size'],
                           anneal_rate=model_params_cv['anneal_rate'],
                           min_temp=lambda_1, verbose=False)
            mf_weights, mf_variances, mf_betas = mf_model.get_weights()

            acc = get_scores(mf_model, x_valsets, y_valsets, single_head)
            ibp_acc = concatenate_results(acc, ibp_acc)

            mf_model.close_session()

        return np.nanmean(ibp_acc)

    # Folder for storing results
    results_folder = "./results/"

    folder = folder_name(results_folder=results_folder,
                         experiment_name=experiment_name,
                         param_bounds=param_bounds,
                         bo_params=bo_params,
                         model_params=model_params)

    os.makedirs(folder, exist_ok=True)

    if os.path.isfile(os.path.join(folder, 'res_max.pkl')):
        print("Loading cached results")
        with open(os.path.join(folder, 'res_max.pkl'), 'rb') as f:
            a = pickle.load(f)
            alpha_opt = a['params']['alpha']
            beta_opt = a['params']['beta']
            lambda_1_opt = a['params']['lambda_1']
            lambda_2_opt = a['params']['lambda_2']
    else:
        print("Running BayesOpt")
        bo = BayesianOptimization(cv_exp, param_bounds)
        bo.maximize()

        with open(os.path.join(folder, 'res_all.pkl'), 'wb') as input_file:
            pickle.dump(bo.res, input_file)

        with open(os.path.join(folder, 'res_max.pkl'), 'wb') as input_file:
            pickle.dump(bo.max, input_file)

        alpha_opt = bo.max['params']['alpha']
        beta_opt = bo.max['params']['beta']
        lambda_1_opt = bo.max['params']['lambda_1']
        lambda_2_opt = bo.max['params']['lambda_2']
    print("alpha_opt: {}".format(alpha_opt))
    print("beta_opt: {}".format(beta_opt))
    print("lambda_1_opt: {}".format(lambda_1_opt))
    print("lambda_2_opt: {}".format(lambda_2_opt))

    ########################################
    ## Experiment with Optimal Parameters ##
    ########################################

    # Run vanilla VCL
    tf.set_random_seed(12)
    np.random.seed(1)

    ibp_acc = np.array([])

    coreset_size = 0
    data_gen = get_data_generator()
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
        bsize = x_train.shape[0] if (model_params['batch_size'] is None) else model_params['batch_size']

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, model_params['hidden_size'], out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, model_params['no_epochs'], bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            mf_betas = None
            ml_model.close_session()

        # Train
        mf_model = MFVI_IBP_NN(in_dim,
                               model_params['hidden_size'],
                               out_dim,
                               x_train.shape[0],
                               num_ibp_samples=model_params['num_ibp_samples'],
                               prev_means=mf_weights,
                               prev_log_variances=mf_variances,
                               prev_betas=mf_betas,
                               alpha0=alpha_opt,
                               beta0=beta_opt,
                               learning_rate=model_params['learning_rate'],
                               lambda_1=lambda_1_opt,
                               lambda_2=lambda_2_opt,
                               no_pred_samples=model_params['pred_samples'],
                               name='opt_{}'.format(experiment_name),
                               beta_1=model_params['beta_1'],
                               beta_2=model_params['beta_2'],
                               beta_3=model_params['beta_3'],
                               epsilon=0.1)

        mf_model.train(x_train, y_train, head, model_params['no_epochs'],
                       model_params['batch_size'],
                       anneal_rate=model_params['anneal_rate'],
                       min_temp=lambda_1_opt, verbose=False)

        mf_weights, mf_variances, mf_betas = mf_model.get_weights()

        acc = get_scores(mf_model, x_testsets, y_testsets, single_head)
        ibp_acc = concatenate_results(acc, ibp_acc) # first col is performance on the first task, second col on the second etc.

        mf_model.close_session()

    ####################################
    ## Run constraited VCL comparison ##
    ####################################

    # Run Vanilla VCL
    tf.reset_default_graph()
    tf.set_random_seed(12)
    np.random.seed(1)
    hidden_size = [5]
    coreset_size = 0
    batch_size = 128
    no_epochs = 500

    data_gen = get_data_generator()
    vcl_result1 = run_vcl(hidden_size, no_epochs, data_gen,
                         lambda a: a, coreset_size, batch_size, single_head, val=True, verbose=False)

    data_gen = get_data_generator()
    hidden_size = [10]
    vcl_result2 = run_vcl(hidden_size, no_epochs, data_gen,
                         lambda a: a, coreset_size, batch_size, single_head, val=True, verbose=False)

    data_gen = get_data_generator()
    hidden_size = [50]
    vcl_result3 = run_vcl(hidden_size, no_epochs, data_gen,
                          lambda a: a, coreset_size, batch_size, single_head, val=True, verbose=False)

    _ibp_acc = np.nanmean(ibp_acc, 1)
    _vcl_result1 = np.nanmean(vcl_result1, 1)
    _vcl_result2 = np.nanmean(vcl_result2, 1)
    _vcl_result3 = np.nanmean(vcl_result3, 1)

    with open(os.path.join(folder, 'res.pkl'), 'wb') as input_file:
        pickle.dump({'vcl_ibp': ibp_acc,
                     'vcl_h5': vcl_result1,
                     'vcl_h10': vcl_result2,
                     'vcl_h50': vcl_result3}, input_file)

    fig = plt.figure(figsize=(7, 3))
    ax = plt.gca()
    plt.plot(np.arange(len(_ibp_acc)) + 1, _ibp_acc, label='VCL + IBP', marker='o')
    plt.plot(np.arange(len(_vcl_result1)) + 1, _vcl_result1, label='VCL h5', marker='o')
    plt.plot(np.arange(len(_vcl_result2)) + 1, _vcl_result2, label='VCL h10', marker='o')
    plt.plot(np.arange(len(_vcl_result3)) + 1, _vcl_result3, label='VCL h50', marker='o')
    ax.set_xticks(range(1, len(_ibp_acc) + 1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()
    fig.savefig("bo_ibp_vcl_res_{}.png".format(args.tag), bbox_inches='tight')
    plt.close()

    print("Finished running.")


