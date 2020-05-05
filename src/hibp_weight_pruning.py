import numpy as np
import tensorflow as tf
import gzip
import pickle
import os.path
import re
import sys
import argparse
sys.path.extend(['alg/'])
from cla_models_multihead import MFVI_NN, Vanilla_NN
from IBP_BNN_multihead import IBP_BNN
from HIBP_BNN_multihead import HIBP_BNN
from copy import deepcopy


def prune_weights(model, X_test, Y_test, bsize, task_id, xs):
    """ Performs weight pruning.

    Z is at a data level doesn't make sense to introduce this intot he mask over weights which get zeroed
    out. Simlpy running the accuracy over the graph will entail that Z is incorporated into the
    matrix math for the prediction calculations.

    Args:
        model: TF model
        X_test: numpy array
        Y_test: numpy array
        bsize: int
        task_id: int
        xs: numpy array
    :return: cutoffs, accs via naive pruning, accs via snr pruning,
    weight values, sigma values of network
    """

    def reset_weights(pr_mus, pr_sigmas, _mus, _sigmas):
        """ Reset weights of graph to original values
        Args:
            pr_mus: list of tf variables which have been pruned
            pr_sigmas: list of tf variables which have been pruned
            _mus: list of cached mus in numpy
            _sigmas: list of cached sigmas in numpy
        """

        for v, _v in zip(pr_mus, _mus):
            model.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

        for v, _v in zip(pr_sigmas, _sigmas):
            model.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

    def pruning(remove_pct, weightvalues, sigmavalues,
                weights, sigmas, bsize, uncert_pruning=True):
        """ Performs weight pruning experiment
        Args:
            weightvalues: np array of weights
            sigmavalues: np array of sigmas
            weights: list of tf weight variable
            new_weights: list of new tf weight variables which wil
            sigmas: list of tf sigma variables
            uncert_pruning: bool pruning by snr
        """
        if uncert_pruning:
            sorted_STN = np.sort(np.abs(weightvalues) / sigmavalues)

        else:
            sorted_STN = np.sort(np.abs(weightvalues))
        cutoff = sorted_STN[int(remove_pct * len(sorted_STN))]

        # Weights, biases and head weights
        for v, s in zip(weights, sigmas):
            if uncert_pruning:
                snr = tf.abs(v) / tf.exp(0.5*s)
                mask = tf.greater_equal(snr, cutoff)
            else:
                mask = tf.greater_equal(tf.abs(v), cutoff)
            model.sess.run(tf.assign(v, tf.multiply(v, tf.cast(mask, v.dtype))))
            #self.sess.run(tf.assign(s, np.multiply(self.sess.run(s), mask)))  # also apply zero std to weight!!!

        acc, _ = model.prediction_acc(X_test, Y_test, bsize, task_id)
        print("%.2f, %s" % (np.sum(sorted_STN < cutoff) / len(sorted_STN), np.mean(acc)))
        return np.mean(acc)

    # get weights
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    # Get weights from network
    mus_w = []
    mus_b = []
    sigmas_w = []
    sigmas_b = []
    mus_h = [] # weights and biases
    sigmas_h = [] # weights and biases
    for v in weights:
        if re.match("^([w])(_mu_)([0-9]+)(:0)$", v.name):
            mus_w.append(v)
        elif re.match("^([w])(_sigma_)([0-9]+)(:0)$", v.name):
            sigmas_w.append(v)
        elif re.match("^([b])(_mu_)([0-9]+)(:0)$", v.name):
            mus_b.append(v)
        elif re.match("^([b])(_sigma_)([0-9]+)(:0)$", v.name):
            sigmas_b.append(v)
        elif re.match("^([wb])(_mu_h_)([0-9]+)(:0)$", v.name):
            mus_h.append(v)
        elif re.match("^([wb])(_sigma_h_)([0-9]+)(:0)$", v.name):
            sigmas_h.append(v)
        else:
            print("Un-matched: {}".format(v.name))

    acc, neg_elbo = model.prediction_acc(X_test, Y_test, bsize,
                                        task_id)  # z mask for each layer in a list, each Z \in dout
    print("test acc: {}".format(acc))
    print("test neg_elbo: {}".format(neg_elbo))
    # cache network weights of resetting the network
    _mus_w = [model.sess.run(w) for w in mus_w]
    _sigmas_w = [model.sess.run(w) for w in sigmas_w]
    _mus_b = [model.sess.run(w) for w in mus_b]
    _sigmas_b = [model.sess.run(w) for w in sigmas_b]
    _mus_h = [model.sess.run(w) for w in mus_h]
    _sigmas_h = [model.sess.run(w) for w in sigmas_h]

    weightvalues = np.hstack(np.array([model.sess.run(w).flatten() for w in mus_w + mus_b + mus_h]))
    sigmavalues = np.hstack(np.array([model.sess.run(tf.exp(0.5*s)).flatten() for s in sigmas_w + sigmas_b + sigmas_h]))

    ya = []
    for pct in xs:
        ya.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                          sigmas_w + sigmas_b + sigmas_h, bsize, uncert_pruning=False))

    # reset etc.
    reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
    reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
    reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)
    yb = []
    for pct in xs:
        yb.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                              sigmas_w + sigmas_b + sigmas_h, bsize, uncert_pruning=True))

    reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
    reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
    reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)

    return xs, ya, yb


class MnistGenerator():
    def __init__(self, val=False):
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

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def task(self):
        # Retrieve train data
        x_train = deepcopy(self.X_train)
        y_train = np.eye(10)[self.Y_train]

        # Retrieve test data
        x_test = deepcopy(self.X_test)
        y_test = np.eye(10)[self.Y_test]

        if self.val:
            x_val = deepcopy(self.X_val)
            y_val = np.eye(10)[self.Y_val]
            return x_train, y_train, x_test, y_test, x_val, y_val

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_baselines', action='store_true',
                        default=False,
                        dest='run_baselines',
                        help='Whether to run MFVI baselines.')
    parser.add_argument('--hibp', action='store_true',
                        default=False,
                        dest='hibp',
                        help='Whether to use HIBP.')
    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')
    parser.add_argument('--no_ibp', action='store_true',
                        default=False,
                        dest='no_ibp',
                        help='Whether not to run ibp.')
    parser.add_argument('--runs', action='store',
                        dest='runs',
                        default=1,
                        type=int,
                        help='Number runs to perform.')
    parser.add_argument('--log_dir', action='store',
                        dest='log_dir',
                        default='logs',
                        help='TB log directory.')
    args = parser.parse_args()

    print('tag                    = {!r}'.format(args.tag))
    print('hibp                   = {!r}'.format(args.hibp))
    print('run_baselines          = {!r}'.format(args.run_baselines))
    print('no_ibp                 = {!r}'.format(args.no_ibp))
    print('runs                   = {!r}'.format(args.runs))
    print('log_dir                = {!r}'.format(args.log_dir))

    hidden_size = [400, 400]
    batch_size = 128
    no_epochs = 200
    seeds = [1, 2, 3, 4, 5]
    np.random.seed(1)
    xs = np.append(0.1 * np.array(range(10)), [0.95, 0.96, 0.97, 0.98, 0.99, 0.992, 0.995, 0.997, 0.999])

    ###########
    ## H-IBP ##
    ###########

    # IBP params
    alpha0 = 4.2
    beta0 = 1.0
    lambda_1 = 0.7  # posterior
    lambda_2 = 0.7  # prior
    alpha = 1.0
    # Gaussian params
    prior_mean = 0.0
    prior_var = 0.7
    val = False
    ya_ibp_all = np.zeros((args.runs, len(xs)))
    yb_ibp_all = np.zeros((args.runs, len(xs)))

    if not args.no_ibp:
        for i in range(args.runs):
            tf.set_random_seed(seeds[i])
            data_gen = MnistGenerator()
            single_head=True
            in_dim, out_dim = data_gen.get_dims()
            x_testsets, y_testsets = [], []
            task_id=0

            tf.reset_default_graph()
            if val:
                x_train, y_train, x_test, y_test, _, _ = data_gen.task()
            else:
                x_train, y_train, x_test, y_test = data_gen.task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)

            # Set the readout head to train
            head = 0 if single_head else task_id
            bsize = x_train.shape[0] if (batch_size is None) else batch_size

            # Train network with maximum likelihood to initialize first model
            if task_id == 0:
                ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
                ml_model.train(x_train, y_train, task_id, 100, bsize)
                mf_weights = ml_model.get_weights()
                mf_variances = None
                mf_betas = None
                ml_model.close_session()

            if args.hibp:
                model = HIBP_BNN(alphas=[alpha]*len(hidden_size),
                                 input_size=in_dim,
                                 hidden_size=hidden_size,
                                 output_size=out_dim,
                                 training_size=x_train.shape[0],
                                 no_pred_samples=100,
                                 num_ibp_samples=10,
                                 no_train_samples=10,
                                 prev_means=mf_weights,
                                 prev_log_variances=mf_variances,
                                 prev_betas=mf_betas,
                                 learning_rate=0.001, learning_rate_decay=0.87,
                                 prior_mean=prior_mean, prior_var=prior_var,
                                 alpha0=alpha0, beta0=beta0,
                                 lambda_1=lambda_1, lambda_2=lambda_2,
                                 tensorboard_dir=args.log_dir,
                                 name='hibp_wp_{0}_run{1}'.format(args.tag, i),
                                 tb_logging=False,
                                 output_tb_gradients=False,
                                 use_local_reparam=False,
                                 implicit_beta=True)
            else:
                model = IBP_BNN(in_dim, hidden_size, out_dim,
                                x_train.shape[0],
                                no_pred_samples=100,
                                num_ibp_samples=10,
                                no_train_samples=10,
                                prev_means=mf_weights,
                                prev_log_variances=mf_variances,
                                prev_betas=mf_betas,
                                learning_rate=0.001, learning_rate_decay=0.87,
                                prior_mean=prior_mean, prior_var=prior_var,
                                alpha0=alpha0, beta0=beta0,
                                lambda_1=lambda_1, lambda_2=lambda_2,
                                tensorboard_dir=args.log_dir,
                                tb_logging=False,
                                output_tb_gradients=False,
                                name='ibp_wp_{0}_run{1}'.format(args.tag, i),
                                use_local_reparam=False,
                                implicit_beta=True)
            model.create_model()

            if os.path.isdir(model.log_folder):
                print("Restoring model from {}".format(model.log_folder))
                model.restore(model.log_folder)
            else:
                print("New model, training")
                model.train(x_train, y_train, head, no_epochs, bsize, verbose=False)

            xs, ya_ibp, yb_ibp = prune_weights(model, x_test, y_test, bsize, head, xs)
            ya_ibp_all[i, :] = ya_ibp
            yb_ibp_all[i, :] = yb_ibp

            model.close_session()

    ##########
    ## MFVI ##
    ##########
    ya_all = np.zeros((args.runs, len(xs)))
    yb_all = np.zeros((args.runs, len(xs)))
    #no_epochs = 200
    if args.run_baselines:
        for i in range(args.runs):
            tf.set_random_seed(seeds[i])
            np.random.seed(1)
            data_gen = MnistGenerator()
            single_head=False
            in_dim, out_dim = data_gen.get_dims()
            task_id=0

            tf.reset_default_graph()
            if val:
                x_train, y_train, x_test, y_test, _, _ = data_gen.task()
            else:
                x_train, y_train, x_test, y_test = data_gen.task()

            # Set the readout head to train
            head = 0 if single_head else task_id
            bsize = x_train.shape[0] if (batch_size is None) else batch_size

            # Train network with maximum likelihood to initialize first model
            if task_id == 0:
                ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
                ml_model.train(x_train, y_train, task_id, 100, bsize)
                mf_weights = ml_model.get_weights()
                mf_variances = None
                ml_model.close_session()

            mf_model = MFVI_NN(in_dim, hidden_size, out_dim,
                               x_train.shape[0], no_train_samples=10, no_pred_samples=100,
                               prev_means=mf_weights, prev_log_variances=mf_variances,
                               learning_rate=0.001, learning_rate_decay=0.50,
                               prior_mean=prior_mean, prior_var=prior_var,
                               use_local_reparam=False)

            if os.path.isdir(mf_model.log_folder):
                print("Restoring model from {}".format(mf_model.log_folder))
                mf_model.restore(mf_model.log_folder)
            else:
                print("New model, training")
                mf_model.train(x_train, y_train, head, no_epochs, bsize)

            xs, ya, yb = prune_weights(mf_model, x_test, y_test, bsize, head, xs)
            ya_all[i, :] = ya
            yb_all[i, :] = yb

            mf_model.close_session()


    with open('results/weight_pruning_{0}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'xs': xs,
                     'ya_nnvi': ya_all,
                     'yb_nnvi': yb_all,
                     'ya_ibp': ya_ibp_all,
                     'yb_ibp': yb_ibp_all}, input_file)