import numpy as np
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
    :return: sample \in [K, batch_size, dout]
    """
    u = tf.random_uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32)
    return tf.exp((1. / (a + eps)) * tf.log(1. - tf.exp((1. / (b + eps)) * tf.log(u)) + eps))

def implicit_beta(a, b, size):
    """
    Returns samples from beta distribution and allows gradients to propagate
    """
    dist = tfd.Beta(a, b)
    samples = dist.sample([size[0], size[1]])
    print("Beta samples shape: {}".format(samples.get_shape()))
    return samples

def reparameterize_beta(a, b, size, ibp=False, log=False, implicit=False):
    """ Returns pi parameters from IBP, cumsum of log v ~ Beta

    :param a: beta a params [dout]
    :param b: beta b params [dout]
    :param size: tuple - size of the beta samples
    :param ibp: bool
    :param log: bool
    :param implicit: bool
    :return: returns truncated variational \pi params \in [K, batch_size, dout]
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

def reparameterize_discrete(log_pis, temp, size):
    """ExpBinConcrete reparam for Bernoulli

    :param log_pis: log variational Bernoulli params \in [K, b, dout]
    :param temp: double
    :return: approx log Bernoulli samples \in [K, b, dout]
    """
    u = tf.random_uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32)
    L = tf.log(u) - tf.log(1. - u)
    return (log_pis + L) / temp

def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y

def get_scores(model, x_testsets, y_testsets, single_head):
    acc = []

    for i in range(len(x_testsets)):

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]

        pred = model.prediction_prob(x_test, head)
        pred_mean = np.mean(pred, axis=0)
        pred_y = np.argmax(pred_mean, axis=1)
        y = np.argmax(y_test, axis=1)
        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)

    #model.close_session()

    return acc

def get_uncertainties(model, x_testsets, y_testsets, single_head, task_id):
    # uncertainties of test set like in Uncertainty in Deep Learning
    # Gal p. 53.
    uncert = []
    for i in range(len(x_testsets)):
        head = 0 if single_head else (task_id if i > task_id else i) # ensures we use the final multi-head model which is available
        x_test, ytext = x_testsets[i], y_testsets[i]
        mi = mutual_information(model, x_test, head)
        uncert.append(mi)
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

def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score

def plot(filename, vcl, rand_vcl, kcen_vcl):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    ax.set_xticks(range(1, len(vcl)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.close()
