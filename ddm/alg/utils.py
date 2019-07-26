import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

eps = 1e-16

def Beta_fn(a, b):
    return tf.exp(tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a+b))

def kl_beta_reparam(a, b, prior_a, prior_b):
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
    kl = tf.cast(kl + tf.log(a * b), tf.float32) + tf.log(Beta_fn(prior_a, prior_b))

    # final term
    kl = kl - tf.cast((b - 1) / b, tf.float32)
    return tf.reduce_sum(kl)

def kl_discrete(log_post, log_prior, log_sample):
    """KL divergence between variational posterior and prior for Bernoulli in test phase

    :param log_prior:
    :param log_post:
    :param z_discrete: Bernoulli samples
    :return: kl
    """
    print("log_post: {}".format(log_post.get_shape()))
    print("log_prior: {}".format(log_prior.get_shape()))
    print("log_sample: {}".format(log_sample.get_shape()))
    pi_post = tf.exp(log_post)
    pi_prior = tf.exp(log_prior)
    z_discrete = tf.exp(log_sample)
    kl_post = z_discrete * tf.log(pi_post + eps) + (1 - z_discrete) * tf.log(1 - pi_post + eps)
    kl_prior = z_discrete * tf.log(pi_prior + eps) + (1 - z_discrete) * tf.log(1 - pi_prior + eps)
    return tf.reduce_sum(kl_post - kl_prior)

def log_density_expconcrete(logpis, logsample, temp):
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

    :param
    """
    log_prior = log_density_expconcrete(log_prior, log_sample, temp_prior)
    log_posterior = log_density_expconcrete(log_post, log_sample, temp)
    kl = log_posterior - log_prior
    return tf.reduce_sum(kl)

def kumaraswamy_sample(a, b, size):
    """

    :param a: beta param a \in [din, dout]
    :param b: beta param b \in [din, dout]
    :return: sample \in [K, din, dout]
    """
    u = tf.random_uniform(shape=size, dtype=tf.float32)
    return tf.exp((1. / (a + eps)) * tf.log(1. - tf.exp((1. / (b + eps)) * tf.log(u)) + eps))

def reparameterize_beta(a, b, size, ibp=False, log=False):
    """ Returns pi parameters from IBP, cumsum of log v ~ Beta

    :param a: beta a params [din, dout]
    :param b: beta b params []
    :param ibp: bool
    :param log: bool
    :return: returns truncated variational \pi params
    """
    print("beta a: {}".format(a.get_shape()))
    print("beta b: {}".format(b.get_shape()))
    v = kumaraswamy_sample(a, b, size)

    if ibp:
        din = size[1]
        K = size[0]
        if din == 1:
            v_term = tf.log(v + eps)
            logpis = tf.cumsum(v_term, axis=2)
        else:
            v_term = tf.log(v+eps)
            _v = tf.cumsum(v_term, axis=1)
            slice = tf.cumsum(_v[:,-1,:-1], axis=1) # K x dout -1
            print("slice: {}".format(slice.get_shape()))
            _slice = tf.expand_dims(slice, 1) # K x 1 x dout - 1
            add = tf.tile(_slice, [1, din, 1]) # K, din, dout - 1
            _add = tf.concat([tf.zeros([K, din, 1]), add], axis=2)
            logpis = _v + _add
    else:
        raise ValueError
    print("logpis: {}".format(logpis.get_shape()))
    if log:
        return logpis
    else:
        return tf.exp(logpis)

def reparameterize_discrete(logpis, temp, size):
    """BinConcrete reparam for Bernoulli

    :param logpis: variational Bernoulli params
    :param temp: double
    :return: approx Bernoulli samples
    """
    u = tf.random_uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32)
    L = tf.log(u) - tf.log(1. - u)
    return (logpis + L) / temp

def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y

def get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size=None):
    from ddm.alg.cla_models_multihead import MFVI_NN, MFVI_IBP_NN
    mf_weights, mf_variances = model.get_weights()
    acc = []

    if single_head:
        if len(x_coresets) > 0:
            x_train, y_train = merge_coresets(x_coresets, y_coresets)
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            final_model = MFVI_NN_IBP(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
            final_model.train(x_train, y_train, 0, no_epochs, bsize)
        else:
            final_model = model

    for i in range(len(x_testsets)):
        if not single_head:
            if len(x_coresets) > 0:
                x_train, y_train = x_coresets[i], y_coresets[i]
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = MFVI_NN_IBP(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
                final_model.train(x_train, y_train, i, no_epochs, bsize)
            else:
                final_model = model

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]

        pred = final_model.prediction_prob(x_test, head)
        pred_mean = np.mean(pred, axis=0)
        pred_y = np.argmax(pred_mean, axis=1)
        y = np.argmax(y_test, axis=1)
        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)

        if len(x_coresets) > 0 and not single_head:
            final_model.close_session()

    if len(x_coresets) > 0 and single_head:
        final_model.close_session()

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
