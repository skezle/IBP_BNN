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

from run_split import SplitMnistImagesGenerator, SplitMnistRandomGenerator, SplitMnistGenerator, SplitMix, SplitCIFAR10Generator
from weight_pruning import MnistGenerator
from vcl import run_vcl_ibp

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

    parser.add_argument('--single_head', action='store_true', default=False, dest='single_head', help='Whether to use a single head.')
    parser.add_argument('--cl2', action='store_true', default=False, dest='cl2', help='Whether to use a perform CL2: domain incremental learning.')
    parser.add_argument('--cl3', action='store_true', default=False, dest='cl3', help='Whether to use a perform CL3: class incremental learning.')
    parser.add_argument('--num_layers', action='store', dest='num_layers', default=1, type=int, help='Number of layers in the NNs.')
    parser.add_argument('--log_dir', action='store', dest='log_dir', default='logs', help='TB log directory.')
    parser.add_argument('--dataset', action='store', dest='dataset', help='Which dataset to choose {normal, random, images, cifar10, mix}.')
    parser.add_argument('--runs', action='store', dest='runs', default=10, type=int, help='Number runs to perform.')
    parser.add_argument('--tag', action='store', dest='tag', help='Tag to use in naming file outputs')
    parser.add_argument('--use_local_reparam', action='store_true', default=False, dest='use_local_reparam', help='Whether to use local reparam.')
    parser.add_argument('--hibp', action='store_true', default=False, dest='hibp', help='Whether to use hibp.')
    parser.add_argument('--K', action='store', dest='K', default=100, type=int, help='Variational truncation param for IBP.')
    parser.add_argument('--optimism', action='store_true', default=False, dest='optimism', help='Whether to use optimism in the face of uncertainty when infering task head for CL2 and CL3.')
    parser.add_argument('--mutual_info', action='store_true', default=False, dest='mutual_info', help='Whether to use predictive entropy or mutual information as a measure of uncertainty for task inference in CL2 and CL3.')
    parser.add_argument('--use_uncert', action='store_true', default=False, dest='use_uncert', help='Whether the uncertainties of the uncertainties to help make choices for inferring CL2 and CL3.')
    parser.add_argument('--batch_entropy', action='store_true', default=False, dest='batch_entropy', help='Whether to use batches when calculating uncertainties for cl2 and cl3.')
    parser.add_argument('--ts_stop_gradients', action='store_true', default=False, dest='ts_stop_gradients', help='Whether to stop gradients using time stamping during training.')
    parser.add_argument('--ts_cutoff', action='store', default=0.5, dest='ts_cutoff', type=float, help='Threshold for the timestamping of Z.')
    parser.add_argument('--ts', action='store_true', default=False, dest='ts', help='Whether to perform timestamping at test time.')
    args = parser.parse_args()

    print('single_head          = {!r}'.format(args.single_head))
    print('cl2                  = {!r}'.format(args.cl2))
    print('cl3                  = {!r}'.format(args.cl3))
    print('num_layers           = {!r}'.format(args.num_layers))
    print('runs                 = {!r}'.format(args.runs))
    print('log_dir              = {!r}'.format(args.log_dir))
    print('dataset              = {!r}'.format(args.dataset))
    print('use_local_reparam    = {!r}'.format(args.use_local_reparam))
    print('hibp                 = {!r}'.format(args.hibp))
    print('K                    = {!r}'.format(args.K))
    print('tag                  = {!r}'.format(args.tag))
    print('ts_stop_gradients    = {!r}'.format(args.ts_stop_gradients))
    print('ts                   = {!r}'.format(args.ts))
    print('ts_cutoff            = {!r}'.format(args.ts_cutoff))

    val = True
    single_head = args.single_head
    task_inf = True if args.cl3 or args.cl2 else False
    assert not (args.cl3 and args.cl2), "Can't have both cl2 and cl3."


    def get_datagen():
        if args.dataset == 'normal':
            data_gen = SplitMnistGenerator(val=val, cl3=args.cl3)
        elif args.dataset == 'random':
            data_gen = SplitMnistRandomGenerator(val=val, cl3=args.cl3)
        elif args.dataset == 'images':
            data_gen = SplitMnistImagesGenerator(val=val, cl3=args.cl3)
        elif args.dataset == 'cifar10':
            data_gen = SplitCIFAR10Generator(val=val, cl3=args.cl3)
        elif args.dataset == 'mix':
            data_gen = SplitMix(val=val, cl3=args.cl3)
        elif args.dataset == 'fmnist':
            data_gen = MnistGenerator(fmnist=True, val=val)
        else:
            raise ValueError('Pick dataset in {normal, random, images, cifar10}')
        return data_gen

    # define hyper parameters

    hyper_param_choices_grid = {'lambda_1': [1/2, 2/3, 3/4, 1, 5/4, 3/2, 7/4, 2, 9/4, 5/2, 11/4, 3],
                                'lambda_2': [1/2, 2/3, 3/4, 1, 5/4, 3/2, 7/4, 2, 9/4, 5/2, 11/4, 3],
                                'a_start': [1, 2, 3, 4, 5],
                                }

    hyper_param_choices_ranges = {'alpha0': [5, 25]}

    fixed_param_choices = {'ibp_samples': 10,
                           'no_pred_samples': 10,
                           'prior_mean': 0.0,
                           'batch_size': 512,
                           'beta0': 1.0,
                           'learning_rate': 3e-4,
                           'learning_rate_decay': 1.0,
                           'prior_var': 0.7,
                           'no_epochs': 1000,
                           'a_step': 1}

    RndSearch = HyperparamOptManager(param_grid=hyper_param_choices_grid,
                                     param_ranges=hyper_param_choices_ranges,
                                     fixed_params=fixed_param_choices,
                                     expt_name='random_search_l{0}_{1}'.format(args.num_layers, args.tag),
                                     root_model_folder='./results/',
                                     network_class=None)

    RndSearch.load_results()

    num_tasks = get_datagen().max_iter
    test_runs = 5
    vcl_ibp_accs = np.zeros((2, test_runs, num_tasks, num_tasks))  # 2 for cl1 and cl2 results
    all_Zs, all_uncerts, time_stamps = [], [], []
    beta_1 = 1
    hidden_size = [args.K] * args.num_layers
    batch_size_entropy = 1500 if args.batch_entropy else None
    coreset_size = 0
    coreset_method = lambda a: a
    val = True
    for i in range(args.runs):
        thetas = RndSearch.get_next_parameters()
        print("thetas: {}".format(thetas))
        name = "ibp_rs_split_{0}_run{1}_{2}".format(args.dataset, i + 1, args.tag)
        data_gen = get_datagen()
        hidden_size = [args.K] * args.num_layers
        # Z matrix for each task is output
        # This is overwritten for each run
        if args.dataset == 'mix':
            alpha = [i for i in range(thetas['a_start'], thetas['a_start'] + thetas['a_step']*(num_tasks//2), thetas['a_step'])]
            alpha = [item for item in alpha for i in range(2)]
        else:
            alpha = [i for i in range(thetas['a_start'], thetas['a_start'] + thetas['a_step']*num_tasks, thetas['a_step'])]
        print("alpha: {}".format(alpha))
        ibp_acc, Zs, uncerts, stamp = run_vcl_ibp(hidden_size=hidden_size, alpha=alpha,
                                                  no_epochs=[int(thetas['no_epochs'])] * num_tasks if int(thetas['no_epochs']) > 600 else [int(
                                                      int(thetas['no_epochs']) * 1.2)] + [int(thetas['no_epochs'])] * (num_tasks - 1),
                                                  data_gen=data_gen, coreset_method=coreset_method,
                                                  coreset_size=coreset_size,
                                                  name=name, val=val, run_val_set=True, batch_size=int(thetas['batch_size']),
                                                  single_head=args.single_head, task_inf=task_inf,
                                                  prior_mean=float(thetas['prior_mean']), prior_var=float(thetas['prior_var']),
                                                  alpha0=float(thetas['alpha0']),
                                                  beta0=float(thetas['beta0']), lambda_1=float(thetas['lambda_1']), lambda_2=float(thetas['lambda_2']),
                                                  learning_rate=[float(thetas['learning_rate'])] * num_tasks,
                                                  learning_rate_decay=float(thetas['learning_rate_decay']),
                                                  no_pred_samples=int(thetas['no_pred_samples']), ibp_samples=int(thetas['ibp_samples']),
                                                  log_dir=args.log_dir,
                                                  use_local_reparam=args.use_local_reparam,
                                                  implicit_beta=True, hibp=args.hibp, beta_1=beta_1,
                                                  optimism=args.optimism,
                                                  pred_ent=False if args.mutual_info else True,
                                                  use_uncert=args.use_uncert, batch_size_entropy=batch_size_entropy,
                                                  ts_stop_gradients=args.ts_stop_gradients, ts=args.ts,
                                                  ts_cutoff=args.ts_cutoff)

        # best score is a loss which is defined to be minimised over, hence want to minimise the negative acc
        if task_inf:
            acc = ibp_acc[1]
        else:
            acc = ibp_acc[0]
        _ = RndSearch.update_score(thetas, -np.nanmean(acc), model=None, sess=None)  # rewards act like the inverse of a loss

    # run final VCL + IBP with opt parameters
    thetas_opt = RndSearch.get_best_params()
    print("Theta opt: {}".format(thetas_opt))
    seed=100
    for i in range(test_runs):
        s = seed + i
        tf.compat.v1.set_random_seed(s)
        data_gen = get_datagen()
        name = "ibp_rs_opt_split_{0}_{1}_run{2}".format(args.dataset, args.tag, i+1)
        if args.dataset == 'mix':
            alpha = [i for i in range(int(thetas_opt['a_start']), int(thetas_opt['a_start']) + int(thetas_opt['a_step'])*(num_tasks//2), int(thetas_opt['a_step']))]
            alpha = [item for item in alpha for i in range(2)]
        else:
            alpha = [i for i in range(int(thetas_opt['a_start']), int(thetas_opt['a_start']) + int(thetas_opt['a_step'])*num_tasks, int(thetas_opt['a_step']))]
        ibp_acc, Zs, uncerts, stamp = run_vcl_ibp(hidden_size=hidden_size, alpha=alpha,
                                                  no_epochs=[int(thetas_opt['no_epochs'])] * num_tasks if int(thetas_opt['no_epochs']) > 600 else [int(
                                                      thetas_opt['no_epochs'] * 1.2)] + [int(thetas_opt['no_epochs'])] * (num_tasks - 1),
                                                  data_gen=data_gen, coreset_method=coreset_method,
                                                  coreset_size=coreset_size,
                                                  name=name, val=val, run_val_set=False, batch_size=int(thetas_opt['batch_size']),
                                                  single_head=args.single_head, task_inf=task_inf,
                                                  prior_mean=float(thetas_opt['prior_mean']), prior_var=float(thetas_opt['prior_var']),
                                                  alpha0=float(thetas_opt['alpha0']),
                                                  beta0=float(thetas_opt['beta0']), lambda_1=float(thetas_opt['lambda_1']), lambda_2=float(thetas_opt['lambda_2']),
                                                  learning_rate=[float(thetas_opt['learning_rate'])] * num_tasks,
                                                  learning_rate_decay=float(thetas_opt['learning_rate_decay']),
                                                  no_pred_samples=int(thetas_opt['no_pred_samples']), ibp_samples=int(thetas_opt['ibp_samples']),
                                                  log_dir=args.log_dir,
                                                  use_local_reparam=args.use_local_reparam,
                                                  implicit_beta=True, hibp=args.hibp, beta_1=beta_1,
                                                  optimism=args.optimism,
                                                  pred_ent=False if args.mutual_info else True,
                                                  use_uncert=args.use_uncert, batch_size_entropy=batch_size_entropy,
                                                  ts_stop_gradients=args.ts_stop_gradients, ts=args.ts,
                                                  ts_cutoff=args.ts_cutoff, seed=s)

        all_Zs.append(Zs)
        all_uncerts.append(uncerts)
        time_stamps.append(stamp)
        vcl_ibp_accs[0, i, :, :] = ibp_acc[0]  # task known
        vcl_ibp_accs[1, i, :, :] = ibp_acc[1]  # task infered
    print("Opt test acc a: {0:.3f}, b: {1:.3f}".format(np.nanmean(vcl_ibp_accs[0, :, :, :]), np.nanmean(vcl_ibp_accs[1, :, :, :])))

    with open('results/split_mnist_{0}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'uncerts_ibp': all_uncerts,
                     'Z': all_Zs,
                     'ts': time_stamps,
                     'opt_thetas': thetas_opt}, input_file)

    print("Finished running.")



