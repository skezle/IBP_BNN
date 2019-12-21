import os.path
import tensorflow as tf
import numpy as np
from copy import deepcopy

from training_utils import kl_beta_reparam, kl_beta_implicit, kl_discrete, kl_concrete
from utils import child_stick_breaking_probs, global_stick_breaking_probs, reparameterize_discrete, implicit_beta
from IBP_BNN_multihead import IBP_BNN

# TODO: hidden state sizes and variational truncation params are a bit awkwardly defined

""" Bayesian Neural Network with VI approximation + IBP """
class HIBP_BNN(IBP_BNN):
    def __init__(self, alphas, input_size, hidden_size, output_size, training_size,
                 no_train_samples=10, no_pred_samples=100, num_ibp_samples=10, prev_means=None, prev_log_variances=None,
                 prev_betas=None, learning_rate=0.001,
                 prior_mean=0, prior_var=1, alpha0=5., beta0=1., lambda_1=1., lambda_2=1.,
                 tensorboard_dir='logs', name='ibp', tb_logging=True, output_tb_gradients=False,
                 beta_1=1.0, beta_2=1.0, beta_3=1.0, use_local_reparam=True, implicit_beta=False):

        super(HIBP_BNN, self).__init__(input_size, hidden_size, output_size, training_size,
                                       no_train_samples, no_pred_samples, num_ibp_samples, prev_means,
                                       prev_log_variances, prev_betas, learning_rate, prior_mean,
                                       prior_var, alpha0, beta0, lambda_1, lambda_2,
                                       tensorboard_dir, name, tb_logging, output_tb_gradients, beta_1, beta_2,
                                       beta_3, use_local_reparam, implicit_beta)
        self.alphas = alphas # hyper-param for each child IBP

    def create_model(self):
        m, v, self.size = self.create_weights(self.input_size, self.hidden_size, self.output_size, self.prev_means, self.prev_log_variances)
        self.W_m, self.b_m, self.W_last_m, self.b_last_m = m[0], m[1], m[2], m[3]
        self.W_v, self.b_v, self.W_last_v, self.b_last_v = v[0], v[1], v[2], v[3]

        betas = self.create_betas(self.prev_betas, self.hidden_size)
        self.gbeta_a, self.gbeta_b = betas[0], betas[1]

        self.weights = [m, v, betas]
        self.no_layers = len(self.size) - 1

        # used for the calculation of the KL term
        m, v = self.create_prior(self.input_size, self.hidden_size, self.output_size, self.prev_means, self.prev_log_variances,
                                 self.prev_betas, self.prior_mean, self.prior_var)
        self.prior_W_m, self.prior_b_m, self.prior_W_last_m, self.prior_b_last_m = m[0], m[1], m[2], m[3]
        self.prior_W_v, self.prior_b_v, self.prior_W_last_v, self.prior_b_last_v = v[0], v[1], v[2], v[3]

        betas = self.create_prior_betas(self.prev_betas, self.hidden_size)
        self.prior_gbeta_a, self.prior_gbeta_b = betas[0], betas[1]

        self.pred, prior_log_pis_bern, log_pis_bern, z_log_sample = self._prediction(self.x, self.task_idx, self.no_pred_samples)

        self.kl = tf.div(self._KL_term(self.no_pred_samples, prior_log_pis_bern, log_pis_bern, z_log_sample),
                         self.training_size)
        self.ll = self._logpred(self.x, self.y, self.task_idx)
        self.cost = self.kl - self.ll
        self.acc = self._accuracy(self.x, self.y, self.task_idx)

        self.assign_optimizer(self.learning_rate)

        self.saver = tf.train.Saver()

        self.create_summaries()

        self.assign_session()

    # this samples a layer at a time
    def _prediction_layer(self, inputs, task_idx, no_samples):
        """ Outputs a prediction from the IBP BNN

        :param inputs: input tensor
        :param task_idx: int
        :param no_samples: int
        :return: output tensor,
                 prior bernoulli params: list of tensors for each layer
                 posterior bernoulli params: list of tensors for each layer
                 log_z_samples: log samples (prior to sigmoid) from expConcrete distribution (for kl calc) for each layer
        """
        # inputs # [batch, d]
        batch_size = tf.to_int32(tf.shape(inputs)[0])
        no_samples_ibp = self.num_ibp_samples
        act = tf.tile(tf.expand_dims(inputs, 0), [no_samples, 1, 1]) # [1, batch, d] --> [K, batch, d]
        act_local = tf.tile(tf.expand_dims(inputs, 0), [no_samples, 1, 1])
        self.Z = []
        self.vars = []
        self.means = []
        log_z_sample = []
        log_pis = []
        prior_log_pis = []
        gbeta_a = tf.cast(tf.math.softplus(self.gbeta_a) + 0.01, tf.float32)
        gbeta_b = tf.cast(tf.math.softplus(self.gbeta_b) + 0.01, tf.float32)
        prior_gbeta_a = tf.cast(tf.math.softplus(self.prior_gbeta_a) + 0.01, tf.float32)
        prior_gbeta_b = tf.cast(tf.math.softplus(self.prior_gbeta_b) + 0.01, tf.float32)
        self.global_log_pi = global_stick_breaking_probs(gbeta_a, gbeta_b, implicit=self.implicit_beta)
        print("global_log_pi: {}".format(self.global_log_pi.get_shape()))
        self.prior_global_log_pi = global_stick_breaking_probs(prior_gbeta_a, prior_gbeta_b, implicit=self.implicit_beta)
        print("prior_global_log_pi: {}".format(self.prior_global_log_pi.get_shape()))
        for i in range(self.no_layers - 1):
            alpha = self.alphas[i]
            din = self.size[i]
            dout = self.size[i + 1]
            eps_w = tf.random_normal((no_samples, din, dout), 0, 1, dtype=tf.float32)
            eps_b = tf.random_normal((no_samples, 1, dout), 0, 1, dtype=tf.float32)

            # Gaussian re-parameterization
            _weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * self.W_v[i])), self.W_m[i]) # in [K, in, out]
            _biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * self.b_v[i])), self.b_m[i])

            # H-IBP
            prior_log_pi = child_stick_breaking_probs(self.prior_global_log_pi, alpha, size=(no_samples_ibp, batch_size, dout))
            prior_log_pis.append(prior_log_pi)
            self.log_pi = child_stick_breaking_probs(self.global_log_pi, alpha, size=(no_samples_ibp, batch_size, dout))
            print("log pi: {}".format(self.log_pi.get_shape()))
            log_pis.append(self.log_pi)
            # Concrete reparam
            z_log_sample = reparameterize_discrete(self.log_pi, self.lambda_1, size=(no_samples_ibp, batch_size, dout))
            z_discrete = tf.expand_dims(tf.reduce_mean(tf.sigmoid(z_log_sample), axis=0), 0)# (K_ibp, batch_size, dout) --> (1, batch_size, dout)

            self.Z.append(z_discrete)
            self.vars += [tf.exp(0.5 * self.W_v[i]), tf.exp(0.5 * self.b_v[i])]
            self.means += [self.W_m[i], self.b_m[i]]
            log_z_sample.append(z_log_sample)
            # multiplication by IBP mask
            pre = tf.add(tf.einsum('mni,mio->mno', act, _weights), _biases) # m = samples, n = din, i=input d, o=output d
            act = tf.multiply(tf.nn.relu(pre), z_discrete) # [K, din, dout]

            # apply local reparameterisation trick: sample activations before applying the non-linearity
            m_h = tf.add(tf.einsum('mni,io->mno', act_local, self.W_m[i]), self.b_m[i])
            v_h = tf.add(tf.einsum('mni,io->mno', tf.square(act_local), tf.square(tf.exp(0.5 * self.W_v[i]))),
                         tf.square(tf.exp(0.5 * self.b_v[i])))
            pre_local = m_h + tf.sqrt(v_h + 1e-6) * tf.random_normal(tf.shape(v_h), dtype=tf.float32)
            act_local = tf.multiply(tf.nn.relu(pre_local), z_discrete)  # shape (K, batch_size (None, ?), out)

        din = self.size[-2]
        dout = self.size[-1]
        eps_w = tf.random_normal((no_samples, din, dout), 0, 1, dtype=tf.float32)
        eps_b = tf.random_normal((no_samples, 1, dout), 0, 1, dtype=tf.float32)

        Wtask_m = tf.gather(self.W_last_m, task_idx)
        Wtask_v = tf.gather(self.W_last_v, task_idx)
        btask_m = tf.gather(self.b_last_m, task_idx)
        btask_v = tf.gather(self.b_last_v, task_idx)

        _weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * Wtask_v)), Wtask_m)
        _biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * btask_v)), btask_m)
        self.vars += [tf.exp(0.5 * Wtask_v), tf.exp(0.5 * btask_v)]
        self.means += [Wtask_m, btask_m]

        act = tf.expand_dims(act, 3) # [K, din, dout, 1]
        _weights = tf.expand_dims(_weights, 1) # [K, 1, dout, 2]
        pre = tf.add(tf.reduce_sum(act * _weights, 2), _biases)

        # apply local reparam trick to final output layer
        _act_local = tf.expand_dims(act_local, 3)  # (K, batch_size, hidden, 1)
        _Wtask_m = tf.expand_dims(tf.expand_dims(Wtask_m, 0), 1)
        _Wtask_v = tf.expand_dims(tf.expand_dims(Wtask_v, 0), 1)

        m_h = tf.add(tf.reduce_sum(_act_local * _Wtask_m, 2), btask_m)
        v_h = tf.add(tf.reduce_sum(tf.square(_act_local) * tf.square(tf.exp(0.5 * Wtask_v)), 2),
                     tf.square(tf.exp(0.5 * btask_v)))
        pre_local = m_h + tf.sqrt(v_h + 1e-6) * tf.random_normal(tf.shape(v_h), dtype=tf.float32)  # (K, batch_size, out_dim)
        return pre, pre_local, prior_log_pis, log_pis, log_z_sample

    def create_summaries(self):
        """Creates summaries in TensorBoard"""
        with tf.name_scope("summaries"):
            tf.compat.v1.summary.scalar("elbo", self.cost)
            tf.compat.v1.summary.scalar("loglik", self.ll)
            tf.compat.v1.summary.scalar("kl", self.kl)
            tf.compat.v1.summary.scalar("kl_gauss", self.kl_gauss)
            tf.compat.v1.summary.scalar("kl_bern", self.kl_bern_contrib)
            tf.compat.v1.summary.scalar("kl_beta", self.kl_beta_contrib)
            tf.compat.v1.summary.scalar("kl_gbeta", self.kl_global_beta_contrib)
            tf.compat.v1.summary.scalar("acc", self.acc)
            Z_all = tf.reduce_sum([tf.cast(tf.reduce_sum(x), tf.float32) for x in self.Z]) / tf.reduce_sum([tf.cast(tf.size(x), tf.float32) for x in self.Z])
            tf.compat.v1.summary.scalar("Z_av", Z_all)
            tf.compat.v1.summary.histogram("W_mu", tf.concat([tf.reshape(i, [-1]) for i in self.means], 0))
            tf.compat.v1.summary.histogram("W_sigma", tf.concat([tf.reshape(i, [-1]) for i in self.vars], 0))
            tf.compat.v1.summary.histogram("v_beta_a_l", tf.cast(tf.math.softplus(tf.exp(tf.log(self.gbeta_a + 1e-8))) + 0.01, tf.float32))
            tf.compat.v1.summary.histogram("v_beta_b_l", tf.cast(tf.math.softplus(tf.exp(tf.log(self.gbeta_b + 1e-8))) + 0.01, tf.float32))
            tf.compat.v1.summary.histogram("p_beta_a_l",
                                           tf.cast(tf.math.softplus(tf.exp(tf.log(self.prior_gbeta_a + 1e-8))) + 0.01,
                                                   tf.float32))
            tf.compat.v1.summary.histogram("p_beta_b_l",
                                           tf.cast(tf.math.softplus(tf.exp(tf.log(self.prior_gbeta_b + 1e-8))) + 0.01,
                                                   tf.float32))
            for i in range(len(self.Z)):
                # tf.summary.images expects 4-d tensor b x height x width x channels
                print("Z: {}".format(self.Z[i].get_shape()))
                _Z = tf.expand_dims(tf.reduce_mean(self.Z[i], axis=0), 0)[:, :50, :] # removing the samples col, and truncating the number of points to make tb faster (hopefully).
                tf.compat.v1.summary.image("Z_{}".format(i),
                                 tf.expand_dims(_Z, 3),
                                 max_outputs=1)

                tf.compat.v1.summary.histogram("Z_num_latent_factors_{}".format(i),
                                    tf.reduce_sum(tf.squeeze(self.Z[i]), axis=1))

            self.summary_op = tf.summary.merge_all()

    def _KL_term(self, no_samples, prior_log_pis, log_pis, z_log_sample):
        kl = 0
        kl_beta = 0
        kl_bern = 0
        for i in range(self.no_layers - 1):
            din = self.size[i]
            dout = self.size[i + 1]
            # weights
            m, v = self.W_m[i], self.W_v[i]
            m0, v0 = self.prior_W_m[i], self.prior_W_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            # biases
            m, v = self.b_m[i], self.b_v[i]
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            # Child IBP Beta terms
            alpha = self.alphas[i]
            kl_beta += kl_beta_implicit(alpha * tf.exp(self.global_log_pi), alpha*(1-tf.exp(self.global_log_pi)),
                                        alpha * tf.exp(self.prior_global_log_pi), alpha*(1-tf.exp(self.prior_global_log_pi)))
            # Child IBP Bernoulli terms
            kl_bern += tf.cond(self.training,
                               true_fn=lambda: kl_concrete(log_pis[i], prior_log_pis[i], z_log_sample[i], self.lambda_1, self.lambda_2),
                               false_fn=lambda: kl_discrete(log_pis[i], prior_log_pis[i], z_log_sample[i]))

        # contribution from the head networks
        # no IBP layer applied to these weights
        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]
        for i in range(no_tasks):
            # weights
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term
            # biases
            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

        # global beta contrib for HIBP
        gbeta_a = tf.cast(tf.math.softplus(self.gbeta_a) + 0.01, tf.float32)
        gbeta_b = tf.cast(tf.math.softplus(self.gbeta_b) + 0.01, tf.float32)
        prior_gbeta_a = tf.cast(tf.math.softplus(self.prior_gbeta_a) + 0.01, tf.float32)
        prior_gbeta_b = tf.cast(tf.math.softplus(self.prior_gbeta_b) + 0.01, tf.float32)
        self.kl_global_beta_contrib = kl_beta_implicit(gbeta_a, gbeta_b, prior_gbeta_a, prior_gbeta_b)

        self.kl_bern_contrib = self.beta_2*kl_bern
        self.kl_beta_contrib = self.beta_3*kl_beta
        self.kl_gauss = self.beta_1 * kl

        return self.kl_gauss + self.kl_beta_contrib + self.kl_bern_contrib + self.kl_global_beta_contrib

    def create_betas(self, prev_betas, hidden_size):
        # global beta params for H-IBP
        # the variational truncation parameter is defined as dout
        hidden_size = deepcopy(hidden_size)
        dout = hidden_size[-1]
        if prev_betas is None:
            global_beta_a_val = tf.constant(np.log(np.exp(self.alpha0) - 1.), shape=[dout])
            global_beta_b_val = tf.constant(np.log(np.exp(self.beta0) - 1.), shape=[dout])
        else:
            global_beta_a_val = prev_betas[0]
            global_beta_b_val = prev_betas[1]

        gb_a = tf.Variable(global_beta_a_val, name="global_beta_a")
        gb_b = tf.Variable(global_beta_b_val, name="global_beta_b")

        return [gb_a, gb_b]

    def create_weights(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_params = 0
        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []
        W_last_m = []
        b_last_m = []
        W_v = []
        b_v = []
        W_last_v = []
        b_last_v = []
        # no last params for betas as head networks do not contain beta params
        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]
            if prev_weights is None:
                Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
                bi_m_val = tf.truncated_normal([dout], stddev=0.1)
                Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                bi_v_val = tf.constant(-6.0, shape=[dout])
            else:
                Wi_m_val = prev_weights[0][i]
                bi_m_val = prev_weights[1][i]
                if prev_variances is None:
                    Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                    bi_v_val = tf.constant(-6.0, shape=[dout])
                else:
                    Wi_v_val = prev_variances[0][i]
                    bi_v_val = prev_variances[1][i]

            Wi_m = tf.Variable(Wi_m_val, name="w_mu_{}".format(i))
            bi_m = tf.Variable(bi_m_val, name="b_mu_{}".format(i))
            Wi_v = tf.Variable(Wi_v_val, name="w_sigma_{}".format(i))
            bi_v = tf.Variable(bi_v_val, name="b_sigma_{}".format(i))

            W_m.append(Wi_m)
            b_m.append(bi_m)
            W_v.append(Wi_v)
            b_v.append(bi_v)

        # if there are previous tasks
        if prev_weights is not None and prev_variances is not None:
            prev_Wlast_m = prev_weights[2]
            prev_blast_m = prev_weights[3]
            prev_Wlast_v = prev_variances[2]
            prev_blast_v = prev_variances[3]
            no_prev_tasks = len(prev_Wlast_m)
            for i in range(no_prev_tasks):
                W_i_m = prev_Wlast_m[i]
                b_i_m = prev_blast_m[i]
                Wi_m = tf.Variable(W_i_m, name="w_mu_h_{}".format(i))
                bi_m = tf.Variable(b_i_m, name="b_mu_h_{}".format(i))

                W_i_v = prev_Wlast_v[i]
                b_i_v = prev_blast_v[i]
                Wi_v = tf.Variable(W_i_v, name="w_sigma_h_{}".format(i))
                bi_v = tf.Variable(b_i_v, name="b_sigma_h_{}".format(i))

                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)

        din = hidden_size[-2]
        dout = hidden_size[-1]

        # if point estimate is supplied
        # before first task we supply the ML solution as a initializer
        if prev_weights is not None and prev_variances is None:
            Wi_m_val = prev_weights[2][0]
            bi_m_val = prev_weights[3][0]
        else:
            Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
            bi_m_val = tf.truncated_normal([dout], stddev=0.1)
        Wi_v_val = tf.constant(-6.0, shape=[din, dout])
        bi_v_val = tf.constant(-6.0, shape=[dout])

        Wi_m = tf.Variable(Wi_m_val, name="w_mu_h_0")
        bi_m = tf.Variable(bi_m_val, name="b_mu_h_0")
        Wi_v = tf.Variable(Wi_v_val, name="w_sigma_h_0")
        bi_v = tf.Variable(bi_v_val, name="b_sigma_h_0")
        W_last_m.append(Wi_m)
        b_last_m.append(bi_m)
        W_last_v.append(Wi_v)
        b_last_v.append(bi_v)

        return [W_m, b_m, W_last_m, b_last_m], [W_v, b_v, W_last_v, b_last_v], hidden_size

    def create_prior_betas(self, prev_betas, hidden_size):
        # Global beta params for H-IBP
        hidden_size = deepcopy(hidden_size)
        dout = hidden_size[-1]
        if prev_betas is None:
            global_beta_a_val = np.full((dout), self.alpha0)
            global_beta_b_val = np.full((dout), self.beta0)
        else:
            global_beta_a_val = prev_betas[0]
            global_beta_b_val = prev_betas[1]

        return [global_beta_a_val, global_beta_b_val]

    def create_prior(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prev_betas, prior_mean, prior_var):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_params = 0
        no_layers = len(hidden_size) - 1
        W_m = []
        b_m = []
        W_last_m = []
        b_last_m = []
        W_v = []
        b_v = []
        W_last_v = []
        b_last_v = []
        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]
            if prev_weights is not None and prev_variances is not None and prev_betas is not None:
                Wi_m = prev_weights[0][i]
                bi_m = prev_weights[1][i]
                Wi_v = np.exp(prev_variances[0][i])
                bi_v = np.exp(prev_variances[1][i])

            else:
                Wi_m = prior_mean
                bi_m = prior_mean
                Wi_v = prior_var
                bi_v = prior_var

            W_m.append(Wi_m)
            b_m.append(bi_m)
            W_v.append(Wi_v)
            b_v.append(bi_v)

        # if there are previous tasks
        if prev_weights is not None and prev_variances is not None:
            prev_Wlast_m = prev_weights[2]
            prev_blast_m = prev_weights[3]
            prev_Wlast_v = prev_variances[2]
            prev_blast_v = prev_variances[3]
            no_prev_tasks = len(prev_Wlast_m)
            for i in range(no_prev_tasks):
                Wi_m = prev_Wlast_m[i]
                bi_m = prev_blast_m[i]
                Wi_v = np.exp(prev_Wlast_v[i])
                bi_v = np.exp(prev_blast_v[i])

                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)

        din = hidden_size[-2]
        dout = hidden_size[-1]
        Wi_m = prior_mean
        bi_m = prior_mean
        Wi_v = prior_var
        bi_v = prior_var
        W_last_m.append(Wi_m)
        b_last_m.append(bi_m)
        W_last_v.append(Wi_v)
        b_last_v.append(bi_v)

        return [W_m, b_m, W_last_m, b_last_m], [W_v, b_v, W_last_v, b_last_v]

