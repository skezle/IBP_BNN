import pdb
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

eps = 1e-16

def Beta_fn(a, b):
    return tf.exp(tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a+b))

def kl_beta_reparam(_a, _b, _prior_a, _prior_b):
    """ KL divergence between Beta and variational beta post.
    from https://github.com/enalisnick/stick-breaking_dgms

    Unused!

    :param a: variational posterior alpha params \in [din, dout]
    :param b: variational posterior beta params \in [din, dout]
    :param prior_a: Beta alpha params, if task 0 scalar
    :param prior_b: Beta alpha params, if task 0 scalar
    :return: kl for each param \in [din, dout]
    """
    a = tf.cast(_a, tf.float32)
    b = tf.cast(_b, tf.float32)
    prior_a = tf.cast(_prior_a, tf.float32)
    prior_b = tf.cast(_prior_b, tf.float32)
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
    kl = kl + tf.log(a * b) + tf.log(Beta_fn(prior_a, prior_b))

    # final term
    kl = kl - (b - 1) / b
    return tf.reduce_sum(kl)

def kl_beta_implicit(_a, _b, _prior_a, _prior_b):
    """
    Implicit reparameterisation KL divergence

    :param size: tuple
    """
    a = tf.cast(_a, tf.float32)
    b = tf.cast(_b, tf.float32)
    prior_a = tf.cast(_prior_a, tf.float32)
    prior_b = tf.cast(_prior_b, tf.float32)
    variational_posterior = tfd.Beta(a + eps, b + eps, validate_args=True, name='v_post')
    prior = tfd.Beta(prior_a + eps, prior_b + eps, validate_args=True, name='prior')
    kl = variational_posterior.kl_divergence(prior)
    return tf.reduce_sum(kl)

def kl_discrete(log_post, log_prior, log_samples):
    """KL divergence between variational posterior and prior for Bernoulli in test phase

    :param log_prior:
    :param log_post:
    :param log_samples: samples from Concrete distribution, before sigmoid is applied \in (K, din, dout)
    :return: kl
    """
    pi_post = tf.exp(log_post)
    pi_prior = tf.exp(log_prior)
    z_discrete = tf.sigmoid(log_samples)
    kl_post = z_discrete * tf.math.log(pi_post + eps) + (1 - z_discrete) * tf.math.log(1 - pi_post + eps)
    kl_prior = z_discrete * tf.math.log(pi_prior + eps) + (1 - z_discrete) * tf.math.log(1 - pi_prior + eps)
    return tf.reduce_sum(tf.reduce_mean(kl_post - kl_prior, [0, 1]))

def log_density_concrete(logpis, logsample, _temp):
    """ log-density of the ExpConcrete distribution, from
    Maddison et. al. (2017) (right after equation 26) in appendix

    :param logpis: Bernoulli/Concrete params
    :param logsample: samples from Concrete distribution, before sigmoid is applied
    :param _temp: float
    """
    temp = tf.cast(_temp, tf.float32)
    exp_term = logpis - temp * logsample
    log_prob = exp_term + tf.math.log(temp) - 2. * tf.math.softplus(exp_term)
    return log_prob

def kl_concrete(log_post, log_prior, log_sample, temp, temp_prior):
    """KL divergence between the prior and posterior
        inputs are in logit-space

    Samples are drawn for the creation of Z and subsequent averaging. Likewise for the KL
    KL is O(number of Concrete params) = O(dout).

    :param log_post \in [no_samples, batch_size, dout]
    :param log_prior \in [no_samples, batch_size, dout]
    :param log_sample \in [no_samples, batch_size, dout]
    :return kl divergence, scalar
    """
    log_prior = log_density_concrete(log_prior, log_sample, temp_prior)
    log_posterior = log_density_concrete(log_post, log_sample, temp)
    return tf.reduce_sum(tf.reduce_mean(log_posterior - log_prior, [0, 1]))

def kumaraswamy_sample(a, b, size):
    """

    :param a: beta param a \in [dout]
    :param b: beta param b \in [dout]
    :return: sample \in [no_samples, batch_size, dout]
    """
    u = tf.random_uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32)
    return tf.exp((1. / (a + eps)) * tf.math.log(1. - tf.exp((1. / (b + eps)) * tf.math.log(u)) + eps))

def implicit_beta(a, b, size, seed=None):
    """
    Returns samples from beta distribution and allows gradients to propagate
    """
    dist = tfd.Beta(a, b, validate_args=True)
    if len(size) == 0:
        samples = dist.sample(seed=seed) # size of a
    elif len(size) == 1:
        samples = dist.sample([size[0]], seed=seed)
    elif len(size) >= 2:
        samples = dist.sample([size[0], size[1]], seed=seed)  # [size[0], size[1], size of a] --> [size[0], size[1], K/dout]
    else:
        raise ValueError
    return samples

def stick_breaking_probs(a, b, size, ibp=False, log=False, implicit=False, seed=None):
    """ Returns pi parameters from stick-breaking IBP

    :param a: beta a params [dout]
    :param b: beta b params [dout]
    :param size: tuple - size of the beta samples
    :param ibp: bool
    :param log: bool
    :param implicit: bool
    :return: returns truncated variational \pi params \in [no_samples, batch_size, dout]
    """
    v = implicit_beta(a, b, size, seed) if implicit else kumaraswamy_sample(a, b, size)
    if ibp:
        v_term = tf.math.log(v + eps)
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
    v_term = tf.math.log(v + eps)
    logpis = tf.cumsum(v_term, axis=1) # \in [no_samples, dout]
    return logpis

def child_stick_breaking_probs(pis, alpha, size):
    """ Returns pi parameters for child stick-breaking IBPs
    :param pis: stick-breaking probabilities from global IBP
    :param alpha: hyperparameter
    :param size: tuple \in [no_samples, batch_size, dout]
    """
    pis = implicit_beta(alpha * pis + eps, alpha*(1-pis) + eps, size)
    logpis = tf.math.log(pis + eps)
    return logpis

def reparameterize_discrete(log_pis, temp, size, seed=None):
    """ExpBinConcrete reparam for Bernoulli

    :param log_pis: log variational Bernoulli params \in [K, b, dout]
    :param temp: double
    :return: approx log Bernoulli samples \in [K, b, dout]
    """
    u = tf.random.uniform(shape=size, minval=1e-4, maxval=1.-1e-4, dtype=tf.float32, seed=seed)
    L = tf.math.log(u) - tf.math.log(1. - u)
    return (log_pis + L) / temp