"""
Random search over sets of hyperparameters or over hyperparam ranges


Created by limsi on 04/10/2018
"""
import os
import shutil
import pandas as pd
import numpy as np
import argparse
import pickle
import tensorflow as tf

from run_split import SplitMnistBackgroundGenerator, SplitMnistRandomGenerator, SplitMnistGenerator
from run_not import NotMnistGenerator
from vcl import run_vcl_ibp, run_vcl
from visualise import plot_uncertainties, plot_Zs

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

np.random.seed(123)

class HyperparamOptManager:

    def __init__(self,
                 param_grid, # dict of params which are to be selected from sets
                 param_ranges, # dict of params which can vary across range
                 fixed_params, # dict of params which are fixed
                 expt_name, # file name for saving results of hyper params
                 root_model_folder, # dir for file above
                 network_class, # ? None
                 override_w_fixed_params=False):

        self.param_grid = param_grid
        self.param_ranges = param_ranges
        self.expt_name = expt_name

        self.max_tries = 1000
        self.results = pd.DataFrame()
        self.fixed_params = fixed_params
        self.saved_params = pd.DataFrame()
        self.NetworkClass = network_class

        self.best_score = np.Inf
        self.optimal_name = ""

        # Create folder for saving if its not there
        self.hyperparam_folder = os.path.join(root_model_folder, expt_name)

        if not os.path.exists(self.hyperparam_folder):
            os.makedirs(self.hyperparam_folder)

        self.override_w_fixed_params = override_w_fixed_params

    def load_results(self):
        print("Loading results from", self.hyperparam_folder)

        results_file = os.path.join(self.hyperparam_folder, "results.csv")
        params_file = os.path.join(self.hyperparam_folder, "params.csv")

        if os.path.exists(results_file) and os.path.exists(params_file):

            self.results = pd.read_csv(results_file, index_col=0)
            self.saved_params = pd.read_csv(params_file, index_col=0)

            if not self.results.empty:
                self.results.at['loss'] = self.results.loc['loss'].apply(lambda x: float(x))
                self.best_score = self.results.loc['loss'].min()

                is_optimal = self.results.loc['loss'] == self.best_score
                self.optimal_name = self.results.T[is_optimal].index[0]

                # self.results.at['is_optimal'] = self.results.loc['loss'] == self.results.loc['loss'].min()  #self.results.loc['is_optimal'].apply(lambda x: str(x).lower() == "true")

                return True

        return False

    def _get_params_from_name(self, name):
        params = self.saved_params

        selected_params = dict(params[name])

        if self.override_w_fixed_params:
            for k in self.fixed_params:
                if k in selected_params:
                    selected_params[k] = self.fixed_params[k]

        return selected_params

    def get_best_params(self):

        optimal_name = self.optimal_name

        return self._get_params_from_name(optimal_name)

    def clear(self):
        shutil.rmtree(self.hyperparam_folder)
        os.makedirs(self.hyperparam_folder)
        self.results = pd.DataFrame()
        self.saved_params = pd.DataFrame()

    def _check_params(self, params):
        # Check that params are valid
        valid_fields = list(self.param_grid.keys()) + list(self.param_ranges.keys()) + list(self.fixed_params.keys())
        invalid_fields = [k for k in params if k not in valid_fields]
        missing_fields = [k for k in valid_fields if k not in params]

        if invalid_fields:
            raise ValueError("Invalid Fields Found {} - Valid ones are {}".format(invalid_fields,
                                                                                 valid_fields))
        if missing_fields:
            raise ValueError("Missing Fields Found {} - Valid ones are {}".format(missing_fields,
                                                                                  valid_fields))

    def _get_name(self, params):

        self._check_params(params)

        fields = list(params.keys())
        fields.sort()

        return "_".join([str(params[k]) for k in fields])

    def get_next_parameters(self):

        def _get_next():

            # sample random range
            parameters = {k: np.random.choice(self.param_grid[k]) for k in self.param_grid}

            for k in self.param_ranges.keys():
                parameters[k] = np.random.uniform(self.param_ranges[k][0], self.param_ranges[k][1])

            # add fixed params
            for k in self.fixed_params:
                parameters[k] = self.fixed_params[k]

            return parameters

        max_tries = self.max_tries   # prevents from getting stuck

        for i in range(max_tries):

            parameters = _get_next()
            name = self._get_name(parameters)

            if name not in self.results.index:
                return parameters

        # Shouldn't reach here
        raise ValueError("Exceeded max number of hyperparameter searches for {}".format(self.expt_name))

    def update_score(self, parameters, loss, model, sess, info=""):

        if np.isnan(loss):
            loss = np.Inf

        if not os.path.isdir(self.hyperparam_folder):
            os.makedirs(self.hyperparam_folder)

        name = self._get_name(parameters)

        # return whether current parameter set is optimal
        is_optimal = self.results.empty or loss < self.best_score

        # save the first model
        if is_optimal:
            print("Optimal model found, updating")
            self.best_score = loss
            self.optimal_name = name
            if model is not None:
                model.save(self.hyperparam_folder, sess)

            self.results[name] = pd.Series({'loss': loss, 'info':info})
            self.saved_params[name] = pd.Series(parameters)

            self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
            self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

        return is_optimal


class HyperparamErrorManager(HyperparamOptManager):

    def __init__(self,
                 param_grid,  # dict of params which are to be selected from sets
                 param_ranges,
                 fixed_params,
                 expt_name,
                 root_model_folder,
                 network_class,
                 override_w_fixed_params=False):

        super().__init__(param_grid,
                 param_ranges,
                 fixed_params,
                 expt_name,
                 root_model_folder,
                 network_class,
                 override_w_fixed_params)

        self.error_locations = []

    def load_results(self):

        success = super().load_results()

        if success:

            results = self.results.loc['loss'][self.results.loc['loss'] == np.Inf]

            self.error_locations = list(results.index)

            self.results = self.results[[s for s in list(self.results.columns) if s not in self.error_locations]]

        return success

    def get_next_parameters(self):

        name = self.error_locations.pop(0)

        return self._get_params_from_name(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--single_head', action='store_true',
                        default=False,
                        dest='single_head',
                        help='Whether to use a single head.')
    parser.add_argument('--noise', action='store_true',
                        default=False,
                        dest='noise',
                        help='Whether to add noise to not MNIST dataset.')
    parser.add_argument('--num_layers', action='store',
                        dest='num_layers',
                        default=1,
                        type=int,
                        help='Number of layers in the NNs.')
    parser.add_argument('--runs', action='store',
                        dest='runs',
                        default=20,
                        type=int,
                        help='Number iterations of random search.')
    parser.add_argument('--log_dir', action='store',
                        dest='log_dir',
                        default='logs',
                        help='TB log directory.')
    parser.add_argument('--dataset', action='store',
                        dest='dataset',
                        help='Which dataset to choose {normal, noise, background, not}.')
    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')
    parser.add_argument('--use_local_reparam', action='store_true',
                        default=False,
                        dest='use_local_reparam',
                        help='Whether to use local reparam.')
    parser.add_argument('--implicit_beta', action='store_true',
                        default=False,
                        dest='implicit_beta',
                        help='Whether to use reparam for Beta dist.')
    parser.add_argument('--hibp', action='store_true',
                        default=False,
                        dest='hibp',
                        help='Whether to use HIBP.')
    args = parser.parse_args()

    print('single_head            = {!r}'.format(args.single_head))
    print('implicit_beta          = {!r}'.format(args.implicit_beta))
    print('num_layers             = {!r}'.format(args.num_layers))
    print('runs                   = {!r}'.format(args.runs))
    print('log_dir                = {!r}'.format(args.log_dir))
    print('dataset                = {!r}'.format(args.dataset))
    print('use_local_reparam      = {!r}'.format(args.use_local_reparam))
    print('noise                  = {!r}'.format(args.noise))
    print('hibp                   = {!r}'.format(args.hibp))
    print('tag                    = {!r}'.format(args.tag))

    seeds = list(range(10, 10 + 5))
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

    # define data generator
    def get_datagen(val):
        if args.dataset == 'normal':
            data_gen = SplitMnistGenerator(val=val, difficult=False)
        elif args.dataset == 'random':
            data_gen = SplitMnistRandomGenerator(val=val)
        elif args.dataset == 'background':
            data_gen = SplitMnistBackgroundGenerator(val=val)
        elif args.dataset == 'not':
            data_gen = NotMnistGenerator(val =val, noise=args.noise)
        else:
            raise ValueError('Pick dataset in {normal, random, background, not}')
        return data_gen

    # define hyper parameters

    hyper_param_choices_grid = {'batch_size': [128, 256, 512]}

    hyper_param_choices_ranges = {'learning_rate': [0.00001, 0.0001],
                                  'alpha0': [1., 5.],
                                  'beta0': [0.5, 1.],
                                  'lambda_1': [0.1, 1.],
                                  'lambda_2': [0.1, 1.],
                                  'prior_var': [0.001, 1.],
                                  'alpha':[1., 5.]}

    fixed_param_choices = {'ibp_samples': 10,
                           'no_pred_samples': 100,
                           'prior_mean': 0.0}

    RndSearch = HyperparamOptManager(param_grid=hyper_param_choices_grid,
                                     param_ranges=hyper_param_choices_ranges,
                                     fixed_params=fixed_param_choices,
                                     expt_name='random_search_l{0}_{1}'.format(args.num_layers, args.tag),
                                     root_model_folder='./results/',
                                     network_class=None)

    RndSearch.load_results()
    hidden_size = [100] * args.num_layers
    no_epochs = 1000
    coreset_size = 0
    val = True
    for i in range(args.runs):
        data_gen = get_datagen(val=val)
        thetas = RndSearch.get_next_parameters()
        name = "ibp_rs_split_{0}_run{1}_{2}".format(args.dataset, i + 1, args.tag)

        ibp_acc, _, _ = run_vcl_ibp(hidden_size=hidden_size, alphas=[thetas['alpha']]*len(hidden_size),
                                    no_epochs=[no_epochs]*num_tasks, data_gen=data_gen,
                                    name=name, val=val, batch_size=thetas['batch_size'],
                                    single_head=args.single_head, prior_mean=thetas['prior_mean'],
                                    prior_var=thetas['prior_var'], alpha0=thetas['alpha0'],
                                    beta0=thetas['beta0'], lambda_1=thetas['lambda_1'],
                                    lambda_2=thetas['lambda_2'], learning_rate=thetas['learning_rate'],
                                    no_pred_samples=thetas['no_pred_samples'],
                                    ibp_samples=thetas['ibp_samples'],
                                    log_dir=args.log_dir, run_val_set=val,
                                    use_local_reparam=args.use_local_reparam,
                                    implicit_beta=args.implicit_beta,
                                    hibp=args.hibp)

        # best score is a loss which is defined to be minimised over, hence want to minimise the negative acc
        _ = RndSearch.update_score(thetas, -np.nanmean(ibp_acc), model=None, sess=None)  # rewards act like the inverse of a loss

    # run final VCL + IBP with opt parameters
    thetas_opt = RndSearch.get_best_params()
    val = False
    for i in range(len(seeds)):
        s = seeds[i]
        tf.set_random_seed(s)
        data_gen = get_datagen(val)
        name = "ibp_rs_opt_split_{0}_{1}_run{2}".format(args.dataset, args.tag, i+1)
        ibp_acc, Zs, uncerts = run_vcl_ibp(hidden_size=hidden_size, alphas=[thetas_opt['alpha']]*len(hidden_size),
                                           no_epochs=[no_epochs]*num_tasks, data_gen=data_gen,
                                           name=name, val=val, batch_size=int(thetas_opt['batch_size']),
                                           single_head=args.single_head, prior_mean=thetas_opt['prior_mean'],
                                           prior_var=thetas_opt['prior_var'], alpha0=thetas_opt['alpha0'],
                                           beta0=thetas_opt['beta0'], lambda_1=thetas_opt['lambda_1'],
                                           lambda_2=thetas_opt['lambda_2'], learning_rate=thetas_opt['learning_rate'],
                                           no_pred_samples=int(thetas_opt['no_pred_samples']),
                                           ibp_samples=int(thetas_opt['ibp_samples']),
                                           log_dir=args.log_dir, run_val_set=False,
                                           use_local_reparam=args.use_local_reparam,
                                           implicit_beta=args.implicit_beta,
                                           hibp=args.hibp)

        all_Zs.append(Zs)
        vcl_ibp_accs[i, :, :] = ibp_acc
        all_ibp_uncerts[i, :, :] = uncerts

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [10] * args.num_layers
        data_gen = get_datagen(val)
        vcl_result_h10, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                          lambda a: a, coreset_size, int(thetas_opt['batch_size']), args.single_head, val=val,
                                          name='vcl_h10_{0}_run{1}'.format(args.tag, i + 1),
                                          log_dir=args.log_dir, use_local_reparam=args.use_local_reparam)
        vcl_h10_accs[i, :, :] = vcl_result_h10
        all_vcl_h10_uncerts[i, :, :] = uncerts

        tf.reset_default_graph()
        hidden_size = [5] * args.num_layers
        data_gen = get_datagen(val)
        vcl_result_h5, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                         lambda a: a, coreset_size, int(thetas_opt['batch_size']), args.single_head, val=val,
                                         name='vcl_h5_{0}_run{1}'.format(args.tag, i + 1),
                                         log_dir=args.log_dir, use_local_reparam=args.use_local_reparam)
        vcl_h5_accs[i, :, :] = vcl_result_h5
        all_vcl_h5_uncerts[i, :, :] = uncerts

        # Run Vanilla VCL
        tf.reset_default_graph()
        hidden_size = [50] * args.num_layers
        data_gen = get_datagen(val)
        vcl_result_h50, uncerts = run_vcl(hidden_size, no_epochs, data_gen,
                                          lambda a: a, coreset_size, int(thetas_opt['batch_size']), args.single_head, val=val,
                                          name='vcl_h50_{0}_run{1}'.format(args.tag, i + 1),
                                          log_dir=args.log_dir, use_local_reparam=args.use_local_reparam)
        vcl_h50_accs[i, :, :] = vcl_result_h50
        all_vcl_h50_uncerts[i, :, :] = uncerts

    tag="ibp_rs_split_{0}_{1}".format(args.dataset, args.tag)
    with open('results/split_mnist_res5_{}.pkl'.format(tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'vcl_h10': vcl_h10_accs,
                     'vcl_h5': vcl_h5_accs,
                     'vcl_h50': vcl_h50_accs,
                     'uncerts_ibp': all_ibp_uncerts,
                     'uncerts_vcl_h5': all_vcl_h5_uncerts,
                     'uncerts_vcl_h10': all_vcl_h10_uncerts,
                     'uncerts_vcl_h50': all_vcl_h50_uncerts,
                     'Z': all_Zs,
                     'opt_params': thetas_opt}, input_file)

    print("Finished running.")



