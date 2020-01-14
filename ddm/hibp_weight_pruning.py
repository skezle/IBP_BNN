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

import matplotlib
import matplotlib.pyplot as plt


class IBP_NN_prune(IBP_BNN):
    def __init__(self, input_size, hidden_size, output_size, training_size, num_ibp_samples=10,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 prev_betas=None, learning_rate=0.001, learning_rate_decay=0.87,
                 prior_mean=0, prior_var=1, alpha0=5., beta0=1., lambda_1=1., lambda_2=1.,
                 tensorboard_dir='logs', name='ibp_wp', min_temp=0.5, tb_logging=True,
                 output_tb_gradients=True, use_local_reparam=False, implicit_beta=True):

        super(IBP_NN_prune, self).__init__(input_size, hidden_size, output_size, training_size,
                 no_train_samples=no_train_samples, no_pred_samples=no_pred_samples,
                 num_ibp_samples=num_ibp_samples, prev_means=prev_means, prev_log_variances=prev_log_variances,
                 prev_betas=prev_betas, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                 prior_mean=prior_mean, prior_var=prior_var, alpha0=alpha0, beta0=beta0,
                 lambda_1=lambda_1, lambda_2=lambda_2, tensorboard_dir=tensorboard_dir,
                 name=name, tb_logging=tb_logging, output_tb_gradients=output_tb_gradients,
                 use_local_reparam=use_local_reparam, implicit_beta=implicit_beta)

    def prune_weights(self, X_test, Y_test, task_id):
        """ Performs weight pruning.

        Z is at a data level doesn't make sense to introduce this intot he mask over weights which get zeroed
        out. Simlpy running the accuracy over the graph will entail that Z is incorporated into the
        matrix math for the prediction calculations.

        Args:
            X_test: numpy array
            Y_test: numpy array
            task_id: int
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
                self.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

            for v, _v in zip(pr_sigmas, _sigmas):
                self.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

        def pruning(remove_pct, weightvalues, sigmavalues,
                    weights, sigmas, uncert_pruning=True):
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
                    snr = tf.abs(v) / tf.exp(0.5 * s)
                    mask = tf.greater_equal(snr, cutoff)
                else:
                    mask = tf.greater_equal(tf.abs(v), cutoff)
                self.sess.run(tf.assign(v, tf.multiply(v, tf.cast(mask, v.dtype))))
                # self.sess.run(tf.assign(s, np.multiply(self.sess.run(s), mask)))  # also apply zero std to weight!!!

            accs = []
            for _ in range(10):
                accs.append(self.sess.run(self.acc, {self.x: X_test,
                                                     self.y: Y_test,
                                                     self.task_idx: task_id,
                                                     self.training: False}))
            print("%.2f, %s" % (np.sum(sorted_STN < cutoff) / len(sorted_STN), np.mean(accs)))
            return np.mean(accs)

        # get weights
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        # Get weights from network
        # TODO get head weights and biases
        mus_w = []
        mus_b = []
        sigmas_w = []
        sigmas_b = []
        mus_h = []  # weights and biases
        sigmas_h = []  # weights and biases
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

        acc = self.sess.run(self.acc, {self.x: X_test,
                                       self.y: Y_test,
                                       self.task_idx: task_id,
                                       self.training: False})  # z mask for each layer in a list, each Z \in dout
        print("test acc: {}".format(acc))
        # cache network weights of resetting the network
        _mus_w = [self.sess.run(w) for w in mus_w]
        _sigmas_w = [self.sess.run(w) for w in sigmas_w]
        _mus_b = [self.sess.run(w) for w in mus_b]
        _sigmas_b = [self.sess.run(w) for w in sigmas_b]
        _mus_h = [self.sess.run(w) for w in mus_h]
        _sigmas_h = [self.sess.run(w) for w in sigmas_h]

        weightvalues = np.hstack(np.array([self.sess.run(w).flatten() for w in mus_w + mus_b + mus_h]))
        sigmavalues = np.hstack(
            np.array([self.sess.run(tf.exp(0.5 * s)).flatten() for s in sigmas_w + sigmas_b + sigmas_h]))

        xs = np.append(0.05 * np.array(range(20)), np.array([0.98, 0.99, 0.999]))
        ya_ibp = []
        for pct in xs:
            ya_ibp.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                                  sigmas_w + sigmas_b + sigmas_h, uncert_pruning=False))

        # reset etc.
        reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
        reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
        reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)
        yb_ibp = []
        for pct in xs:
            yb_ibp.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                                  sigmas_w + sigmas_b + sigmas_h, uncert_pruning=True))

        reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
        reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
        reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)

        # return xs, ya, yb, ya_ibp, yb_ibp
        return xs, ya_ibp, yb_ibp


class MFVI_NN_prune(MFVI_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size,
        no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None, learning_rate=0.001,
        prior_mean=0, prior_var=1):

        super(MFVI_NN_prune, self).__init__(input_size, hidden_size, output_size, training_size,
        no_train_samples, no_pred_samples, prev_means, prev_log_variances, learning_rate,
        prior_mean, prior_var)


    def prune_weights(self, X_test, Y_test, task_id):
        """ Performs weight pruning.

        Z is at a data level doesn't make sense to introduce this intot he mask over weights which get zeroed
        out. Simlpy running the accuracy over the graph will entail that Z is incorporated into the
        matrix math for the prediction calculations.

        Args:
            X_test: numpy array
            Y_test: numpy array
            task_id: int
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
                self.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

            for v, _v in zip(pr_sigmas, _sigmas):
                self.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

        def pruning(remove_pct, weightvalues, sigmavalues,
                    weights, sigmas, uncert_pruning=True):
            """ Performs weight pruning experiment
            Args:
                weightvalues: np array of weights
                sigmavalues: np array of sigmas
                weights: list of tf weight variable
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
                self.sess.run(tf.assign(v, tf.multiply(v, tf.cast(mask, v.dtype))))

            accs = []
            for _ in range(10):
                accs.append(self.sess.run(self.acc, {self.x: X_test,
                                                     self.y: Y_test,
                                                     self.task_idx: task_id}))
            print("%.2f, %s" % (np.sum(sorted_STN < cutoff) / len(sorted_STN), np.mean(accs)))
            return np.mean(accs)

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

        # cache network weights of resetting the network
        _mus_w = [self.sess.run(w) for w in mus_w]
        _sigmas_w = [self.sess.run(w) for w in sigmas_w]
        _mus_b = [self.sess.run(w) for w in mus_b]
        _sigmas_b = [self.sess.run(w) for w in sigmas_b]
        _mus_h = [self.sess.run(w) for w in mus_h]
        _sigmas_h = [self.sess.run(w) for w in sigmas_h]

        weightvalues = np.hstack(np.array([self.sess.run(w).flatten() for w in mus_w + mus_b + mus_h]))
        sigmavalues = np.hstack(np.array([self.sess.run(tf.exp(0.5*s)).flatten() for s in sigmas_w + sigmas_b + sigmas_h]))

        xs = np.append(0.05 * np.array(range(20)), np.array([0.98, 0.99, 0.999]))
        # pruning
        ya = []
        for pct in xs:
            ya.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                              sigmas_w + sigmas_b + sigmas_h, uncert_pruning=False))

        # reset etc.
        reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
        reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
        reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)
        yb = []
        for pct in xs:
            yb.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                                  sigmas_w + sigmas_b + sigmas_h, uncert_pruning=True))

        reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
        reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
        reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)

        return xs, ya, yb


class HIBP_BNN_prune(HIBP_BNN):
    def __init__(self, alphas, input_size, hidden_size,
                 output_size, training_size, num_ibp_samples=10,
                 no_train_samples=10, no_pred_samples=100,
                 prev_means=None, prev_log_variances=None,
                 prev_betas=None, learning_rate=0.0001,
                 learning_rate_decay=0.87,
                 prior_mean=0, prior_var=1, alpha0=5.,
                 beta0=1., lambda_1=1., lambda_2=1.,
                 tensorboard_dir='logs', name='hibp_wp',
                 tb_logging=True,
                 output_tb_gradients=True, beta_1=1.0, beta_2=1.0,
                 beta_3=1.0, use_local_reparam=False, implicit_beta=True):

        super(HIBP_BNN_prune, self).__init__(alphas=alphas,
                input_size=input_size, hidden_size=hidden_size,
                output_size=output_size,
                training_size=training_size, num_ibp_samples=num_ibp_samples,
                 no_train_samples=no_train_samples, no_pred_samples=no_pred_samples,
                 prev_means=prev_means, prev_log_variances=prev_log_variances,
                 prev_betas=prev_betas, learning_rate=learning_rate,
                 learning_rate_decay=learning_rate_decay,
                 prior_mean=prior_mean, prior_var=prior_var, alpha0=alpha0,
                 beta0=beta0, lambda_1=lambda_1, lambda_2=lambda_2,
                 tensorboard_dir=tensorboard_dir, name=name,
                 tb_logging=tb_logging,
                 output_tb_gradients=output_tb_gradients, beta_1=beta_1,
                 beta_2=beta_2, beta_3=beta_3, use_local_reparam=use_local_reparam,
                 implicit_beta=implicit_beta)


    def prune_weights(self, X_test, Y_test, task_id):
        """ Performs weight pruning.

        Z is at a data level doesn't make sense to introduce this intot he mask over weights which get zeroed
        out. Simlpy running the accuracy over the graph will entail that Z is incorporated into the
        matrix math for the prediction calculations.

        Args:
            X_test: numpy array
            Y_test: numpy array
            task_id: int
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
                self.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

            for v, _v in zip(pr_sigmas, _sigmas):
                self.sess.run(tf.assign(v, tf.cast(_v, v.dtype)))

        def pruning(remove_pct, weightvalues, sigmavalues,
                    weights, sigmas, uncert_pruning=True):
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
                self.sess.run(tf.assign(v, tf.multiply(v, tf.cast(mask, v.dtype))))
                #self.sess.run(tf.assign(s, np.multiply(self.sess.run(s), mask)))  # also apply zero std to weight!!!

            accs = []
            for _ in range(10):
                accs.append(self.sess.run(self.acc, {self.x: X_test,
                                                     self.y: Y_test,
                                                     self.task_idx: task_id,
                                                     self.training: False}))
            print("%.2f, %s" % (np.sum(sorted_STN < cutoff) / len(sorted_STN), np.mean(accs)))
            return np.mean(accs)

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

        acc = self.sess.run(self.acc, {self.x: X_test,
                                       self.y: Y_test,
                                       self.task_idx: task_id,
                                       self.training: False}) # z mask for each layer in a list, each Z \in dout
        print("test acc: {}".format(acc))
        # cache network weights of resetting the network
        _mus_w = [self.sess.run(w) for w in mus_w]
        _sigmas_w = [self.sess.run(w) for w in sigmas_w]
        _mus_b = [self.sess.run(w) for w in mus_b]
        _sigmas_b = [self.sess.run(w) for w in sigmas_b]
        _mus_h = [self.sess.run(w) for w in mus_h]
        _sigmas_h = [self.sess.run(w) for w in sigmas_h]

        weightvalues = np.hstack(np.array([self.sess.run(w).flatten() for w in mus_w + mus_b + mus_h]))
        sigmavalues = np.hstack(np.array([self.sess.run(tf.exp(0.5*s)).flatten() for s in sigmas_w + sigmas_b + sigmas_h]))

        xs = np.append(0.05 * np.array(range(20)), np.array([0.98, 0.99, 0.999]))
        ya_ibp = []
        for pct in xs:
            ya_ibp.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                              sigmas_w + sigmas_b + sigmas_h, uncert_pruning=False))

        # reset etc.
        reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
        reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
        reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)
        yb_ibp = []
        for pct in xs:
            yb_ibp.append(pruning(pct, weightvalues, sigmavalues, mus_w + mus_b + mus_h,
                                  sigmas_w + sigmas_b + sigmas_h, uncert_pruning=True))

        reset_weights(mus_w, sigmas_w, _mus_w, _sigmas_w)
        reset_weights(mus_b, sigmas_b, _mus_b, _sigmas_b)
        reset_weights(mus_h, sigmas_h, _mus_h, _sigmas_h)

        #return xs, ya, yb, ya_ibp, yb_ibp
        return xs, ya_ibp, yb_ibp


class MnistGenerator():
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

    def task(self):
        # Retrieve train data
        x_train = deepcopy(self.X_train)
        y_train = np.eye(10)[self.Y_train]

        # Retrieve test data
        x_test = deepcopy(self.X_test)
        y_test = np.eye(10)[self.Y_test]

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
    args = parser.parse_args()

    print('tag                    = {!r}'.format(args.tag))
    print('hibp                   = {!r}'.format(args.hibp))
    print('run_baselines           = {!r}'.format(args.run_baselines))

    hidden_size = [200, 200]
    batch_size = 512
    no_epochs = 500
    runs = 5
    seeds = [1,2,3,4,5]
    np.random.seed(1)
    xs = np.append(0.05 * np.array(range(20)), np.array([0.98, 0.99, 0.999]))

    ###########
    ## H-IBP ##
    ###########
    ya_ibp_all = np.zeros((runs, len(xs)))
    yb_ibp_all = np.zeros((runs, len(xs)))

    for i in range(runs):
        tf.set_random_seed(seeds[i])
        coreset_size = 0
        data_gen = MnistGenerator()
        single_head=False
        in_dim, out_dim = data_gen.get_dims()
        x_testsets, y_testsets = [], []
        task_id=0

        tf.reset_default_graph()
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

        # Train on non-coreset data
        if args.hibp:
            model = HIBP_BNN_prune(alphas=[1.]*len(hidden_size),
                                   input_size=in_dim,
                                   hidden_size=hidden_size,
                                   output_size=out_dim,
                                   training_size=x_train.shape[0],
                                   no_pred_samples=100,
                                   num_ibp_samples=10, prev_means=mf_weights,
                                   prev_log_variances=mf_variances,
                                   prev_betas=mf_betas,
                                   learning_rate=0.001, learning_rate_decay=0.87,
                                   prior_mean=0.0, prior_var=0.1,
                                   alpha0=5.0, beta0=1.0,
                                   lambda_1=1.0, lambda_2=1.0,
                                   tensorboard_dir='logs_wp',
                                   name='hibp_wp_mnist_new_run{0}'.format(i),
                                   use_local_reparam=False, implicit_beta=True)
        else:
            model = IBP_NN_prune(in_dim, hidden_size, out_dim,
                                 x_train.shape[0],
                                 no_pred_samples=100,
                                 num_ibp_samples=10,
                                 prev_means=mf_weights,
                                 prev_log_variances=mf_variances,
                                 prev_betas=mf_betas,
                                 learning_rate=0.001, learning_rate_decay=0.87,
                                 prior_mean=1.0, prior_var=0.1,
                                 alpha0=5.0, beta0=1.0,
                                 lambda_1=1.0, lambda_2=1.0,
                                 tensorboard_dir='logs_wp',
                                 name='ibp_wp_mnist_new_run{0}'.format(i),
                                 use_local_reparam=False, implicit_beta=True)
        model.create_model()

        if os.path.isdir(model.log_folder):
            print("Restoring model from {}".format(model.log_folder))
            model.restore(model.log_folder)
        else:
            print("New model, training")
            model.train(x_train, y_train, head, no_epochs, bsize)

        xs, ya_ibp, yb_ibp = model.prune_weights(x_test, y_test, head)
        ya_ibp_all[i, :] = ya_ibp
        yb_ibp_all[i, :] = yb_ibp

        model.close_session()

    ##########
    ## MFVI ##
    ##########
    ya_all = np.zeros((runs, len(xs)))
    yb_all = np.zeros((runs, len(xs)))
    if args.run_baselines:
        for i in range(runs):
            tf.set_random_seed(seeds[i])
            np.random.seed(1)
            coreset_size = 0
            data_gen = MnistGenerator()
            single_head=False
            in_dim, out_dim = data_gen.get_dims()
            task_id=0

            tf.reset_default_graph()
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

            # Train on non-coreset data
            mf_model = MFVI_NN_prune(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights,
                                     prev_log_variances=mf_variances)

            mf_model.train(x_train, y_train, head, no_epochs, bsize)

            xs, ya, yb = mf_model.prune_weights(x_test, y_test, head)
            ya_all[i, :] = ya
            yb_all[i, :] = yb

            mf_model.close_session()

        ###########
        ## Plots ##
        ###########
        _ibp_ya_mean = np.mean(ya_ibp_all, axis=0)
        _ibp_ya_std = np.std(ya_ibp_all, axis=0)
        _ibp_yb_mean = np.mean(yb_ibp_all, axis=0)
        _ibp_yb_std = np.std(yb_ibp_all, axis=0)
        _ya_mean = np.mean(ya_all, axis=0)
        _ya_std = np.std(ya_all, axis=0)
        _yb_mean = np.mean(yb_all, axis=0)
        _yb_std = np.std(yb_all, axis=0)

        fig_size = (6, 5)

        set_y_axis = False

        lw = 2

        grid_color = '0.1'
        grid_lw = 0.2

        title_size = 16
        label_size = 22
        tick_size = 20
        legend_size = 22

        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        ax.plot(xs, _ibp_ya_mean,linewidth=lw, color='b')
        ax.fill_between(xs,
                       [x - y for x, y in zip(_ibp_ya_mean, _ibp_ya_std)],
                       [x + y for x, y in zip(_ibp_ya_mean, _ibp_ya_std)],
                       alpha=0.3, color='b')

        ax.plot(xs, _ibp_yb_mean, linewidth=lw, color='g')
        ax.fill_between(xs,
                       [x - y for x, y in zip(_ibp_yb_mean, _ibp_yb_std)],
                       [x + y for x, y in zip(_ibp_yb_mean, _ibp_yb_std)],
                       alpha=0.3, color='g')

        ax.plot(xs, _ya_mean,linewidth=lw, color='r')
        ax.fill_between(xs,
                       [x - y for x, y in zip(_ya_mean, _ya_std)],
                       [x + y for x, y in zip(_ya_mean, _ya_std)],
                       alpha=0.3, color='r')

        ax.plot(xs, _yb_mean, linewidth=lw, color='c')
        ax.fill_between(xs,
                       [x - y for x, y in zip(_yb_mean, _yb_std)],
                       [x + y for x, y in zip(_yb_mean, _yb_std)],
                       alpha=0.3, color='c')

        plt.grid(True, which="both", color=grid_color, linewidth=0.1, alpha=0.1)
        ax.set_xlim(0.0, 0.5)
        plt.yticks(fontsize=tick_size)
        x_ticks = np.arange(0.0, 1.5, step=0.5)
        plt.xticks(x_ticks, fontsize=tick_size)
        plt.ticklabel_format(axis='y', style='sci')
        ax = fig.gca()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
        #ax.set_yticklabels([])
        plt.yscale('log')
        plt.legend(["H-IBP $|\mu|$", 'H-IBP snr', '$|\mu|$', 'snr'], fontsize=legend_size, loc='lower left')
        plt.tight_layout()
        plt.xlabel("Cut off", fontsize=legend_size)
        plt.ylabel("Test acc", fontsize=legend_size)
        plt.xlim(0.0, 1.01)
        plt.savefig("plots/weight_pruning_{0}.pdf".format(args.tag), bbox_inches='tight')
        #plt.show()

    xs = np.append(0.05 * np.array(range(20)), np.array([0.98, 0.99, 0.999]))

    with open('results/weight_pruning_runs5_{0}.pkl'.format(args.tag), 'wb') as input_file:
        pickle.dump({'xs': xs,
                     'ya_nnvi': ya_all,
                     'yb_nnvi': yb_all,
                     'ya_ibp': ya_ibp_all,
                     'yb_ibp': yb_ibp_all}, input_file)
