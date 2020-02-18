import numpy as np
import pdb
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

eps = 1e-16

def kumaraswamy_sample(a, b, size):
    """

    :param a: beta param a \in [dout]
    :param b: beta param b \in [dout]
    :return: sample \in [no_samples, batch_size, dout]
    """
    u = tf.random_uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32)
    return tf.exp((1. / (a + eps)) * tf.log(1. - tf.exp((1. / (b + eps)) * tf.log(u)) + eps))

def implicit_beta(a, b, size):
    """
    Returns samples from beta distribution and allows gradients to propagate
    """
    dist = tfd.Beta(a, b, validate_args=True)
    if len(size) == 0:
        samples = dist.sample() # size of a
    elif len(size) == 1:
        samples = dist.sample([size[0]])
    elif len(size) >= 2:
        samples = dist.sample([size[0], size[1]])  # [size[0], size[1], size of a]
    else:
        raise ValueError
    return samples

def stick_breaking_probs(a, b, size, ibp=False, log=False, implicit=False):
    """ Returns pi parameters from stick-breaking IBP

    :param a: beta a params [dout]
    :param b: beta b params [dout]
    :param size: tuple - size of the beta samples
    :param ibp: bool
    :param log: bool
    :param implicit: bool
    :return: returns truncated variational \pi params \in [no_samples, batch_size, dout]
    """
    v = implicit_beta(a, b, size) if implicit else kumaraswamy_sample(a, b, size)
    if ibp:
        v_term = tf.log(v + eps)
        logpis = tf.cumsum(v_term, axis=2)
    else:
        raise ValueError
    if log:
        return logpis
    else:
        return tf.exp(logpis)

def global_stick_breaking_probs(a, b, size, implicit=True):
    """Returns pi parameters from stick-breaking H-IBP

    :param a: beta a params [dout]
    :param b: beta b params [dout]
    :param implicit: bool
    :return: returns truncated variational \pi params
    """
    v = implicit_beta(a, b, size=size) if implicit else kumaraswamy_sample(a, b, size=size)
    v_term = tf.log(v + eps)
    logpis = tf.cumsum(v_term, axis=1) # \in [no_samples, dout]
    return logpis

def child_stick_breaking_probs(pis, alpha, size):
    """ Returns pi parameters for child stick-breaking IBPs
    :param pis: stick-breaking probabilities from global IBP
    :param alpha: hyperparameter
    :param size: tuple \in [no_samples, batch_size, dout]
    """
    pis = implicit_beta(alpha * pis + eps, alpha*(1-pis) + eps, size)
    logpis = tf.log(pis + eps)
    return logpis

def reparameterize_discrete(log_pis, temp, size):
    """ExpBinConcrete reparam for Bernoulli

    :param log_pis: log variational Bernoulli params \in [K, b, dout]
    :param temp: double
    :return: approx log Bernoulli samples \in [K, b, dout]
    """
    u = tf.random_uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32)
    L = tf.log(u) - tf.log(1. - u)
    return (log_pis + L) / temp

def get_uncertainties(model, x_testsets, y_testsets, single_head, task_id, bsize):
    # uncertainties of test set like in Uncertainty in Deep Learning
    # Gal p. 53.
    uncert = []
    for i in range(len(x_testsets)):
        head = 0 if single_head else (task_id if i > task_id else i) # ensures we use the final multi-head model which is available
        x_test, ytext = x_testsets[i], y_testsets[i]
        #mi = mutual_information(model, x_test, head)
        pe = predictive_entropy(model, x_test, head, bsize)
        uncert.append(pe)
    return uncert

def mutual_information(model, x_test, task_idx):
    """ Based off of Yarin's thesis.

    :param model: BNN model object
    :param x_test: test data (n, d)
    :param task_idx: int
    :return:
    """
    mc_samples = [model.prediction_prob(x_test, task_idx) for _ in range(10)]
    mc_samples_ar = np.concatenate(mc_samples, axis=0)
    #pdb.set_trace()
    eps = 1e-16
    expected_p = np.mean(mc_samples_ar, axis=0)
    predictive_entropy = -np.sum(expected_p * np.log(expected_p+eps), axis=-1) # (test_set_size, )
    mc_entropy = np.sum(mc_samples_ar * np.log(mc_samples_ar+eps), axis=-1)
    expected_entropy = -np.mean(mc_entropy, axis=0) # (test_set_size, )
    mi = np.mean(predictive_entropy - expected_entropy)
    return mi

def predictive_entropy(model, x_test, task_idx, bsize):
    mc_samples = model.prediction_prob(x_test, task_idx, bsize)
    # each element of mc_samples is no_pred_samples x batch_size x 2
    # pdb.set_trace()
    mc_samples_ar = np.concatenate(mc_samples, axis=1) # no_pred_samples x test_size x 2
    eps = 1e-16
    expected_p = np.mean(mc_samples_ar, axis=0) # test_size x 2
    predictive_entropy = -np.sum(expected_p * np.log(expected_p + eps), axis=-1)  # (test_set_size, )
    return predictive_entropy # test_size: (n,)

def concatenate_results(score, all_score):
    """New row is added to scores. Matrix: rows are the task
    cols are the evaluations.

    :param score: list
    :param all_score: np array
    :return:
    """
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score

def get_scores(model, x_testsets, y_testsets, batch_size, single_head):
    accs = []
    for i in range(len(x_testsets)):
        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        acc, _ = model.prediction_acc(x_test, y_test, batch_size, head)
        accs.append(acc)
    return accs

def get_Zs(model, x_test, batch_size, task_id):
    Zs = model.prediction_Zs(x_test, batch_size, task_id)
    return Zs
