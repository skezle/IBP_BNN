import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

eps = 1e-16

def Beta_fn(a, b):
    return tf.exp(tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a+b))

def kl_beta_reparam(a, b, prior_a, prior_b):
    """ KL divergence between Beta and variational beta post.
    from https://github.com/enalisnick/stick

    :param a: variational posterior alpha params \in [din, dout]
    :param b: variational posterior beta params \in [din, dout]
    :param prior_a: Beta alpha params, if task 0 scalar
    :param prior_b: Beta alpha params, if task 0 scalar
    :return: kl for each param \in [din, dout]
    """
    kl = 1. / (1 + a * b) * Beta_fn(1. / a, b)
    kl += 1. / (2 + a * b) * Beta_fn(2. / a, b)
    kl += 1. / (3 + a * b) * Beta_fn(3. / a, b)
    kl += 1. / (4 + a * b) * Beta_fn(4. / a, b)
    kl += 1. / (5 + a * b) * Beta_fn(5. / a, b)
    kl += 1. / (6 + a * b) * Beta_fn(6. / a, b)
    kl += 1. / (7 + a * b) * Beta_fn(7. / a, b)
    kl += 1. / (8 + a * b) * Beta_fn(8. / a, b)
    kl += 1. / (9 + a * b) * Beta_fn(9. / a, b)
    kl += 1. / (10 + a * b) * Beta_fn(10. / a, b)
    kl *= (prior_b - 1) * b

    # use another taylor approx for Digamma function
    psi_b_taylor_approx = tf.log(b) - 1. / (2 * b) - 1. / (12 * b ** 2)
    kl += (a - prior_a) / a * (-0.57721 - psi_b_taylor_approx - 1 / b)  # T.psi(self.posterior_b)

    # add normalization constants
    kl = tf.cast(kl + tf.log(a * b), tf.float32) + tf.cast(tf.log(Beta_fn(prior_a, prior_b)), tf.float32)

    # final term
    kl = kl - tf.cast((b - 1) / b, tf.float32)
    return tf.reduce_sum(kl)

def kl_discrete(log_post, log_prior, z_discrete):
    """KL divergence between variational posterior and prior for Bernoulli in test phase

    :param log_prior:
    :param log_post:
    :param z_discrete: Bernoulli samples
    :return: kl
    """

    pi_post = tf.exp(log_post)
    pi_prior = tf.exp(log_prior)
    kl_post = z_discrete * tf.log(pi_post + eps) + (1 - z_discrete) * tf.log(1 - pi_post + eps)
    kl_prior = z_discrete * tf.log(pi_prior + eps) + (1 - z_discrete) * tf.log(1 - pi_prior + eps)
    return tf.reduce_mean(kl_post - kl_prior)

def log_density_concrete(logpis, logsample, temp):
    """ log-density of the ExpConcrete distribution, from
    Maddison et. al. (2017) (right after equation 26) in appendix

    :param logalphas: Bernoulli/Concrete params
    :param logsample: samples from Concrete distribution, before sigmoid is applied
    :param temp: float
    Input logalpha is a logit (alpha is a probability ratio)


    """
    exp_term = logpis - temp * logsample
    log_prob = exp_term + tf.log(temp) - 2. * tf.math.softplus(exp_term)
    return log_prob

def kl_concrete(log_post, log_prior, log_sample, temp, temp_prior):
    """KL divergence between the prior and posterior
        inputs are in logit-space

    :param log_post \in [K, din, dout]
    :param log_prior \in [K, din, dout]
    :param log_sample \in [K, din, dout]
    :return kl divergence, scalar
    """
    log_prior = log_density_concrete(log_prior, log_sample, temp_prior)
    log_posterior = log_density_concrete(log_post, log_sample, temp)
    return tf.reduce_sum(tf.reduce_mean(log_posterior - log_prior, 0))

def kumaraswamy_sample(a, b, size):
    """

    :param a: beta param a \in [din, dout]
    :param b: beta param b \in [din, dout]
    :return: sample \in [K, din, dout]
    """
    u = tf.random_uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32)
    return tf.exp((1. / (a + eps)) * tf.log(1. - tf.exp((1. / (b + eps)) * tf.log(u)) + eps))

def reparameterize_beta(a, b, size, ibp=False, log=False):
    """ Returns pi parameters from IBP, cumsum of log v ~ Beta

    :param a: beta a params [din, dout]
    :param b: beta b params [din, dout]
    :param ibp: bool
    :param log: bool
    :return: returns truncated variational \pi params \in [K, din, dout]
    """
    print("beta a: {}".format(a.get_shape()))
    print("beta b: {}".format(b.get_shape()))
    v = kumaraswamy_sample(a, b, size)

    if ibp:
        v_term = tf.log(v + eps)
        logpis = tf.cumsum(v_term, axis=2)
    else:
        raise ValueError
    print("logpis: {}".format(logpis.get_shape()))
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

    model.close_session()

    return acc

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
