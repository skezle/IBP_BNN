import os.path
import pdb
import tensorflow as tf
import numpy as np
from copy import deepcopy

from training_utils import kl_beta_reparam, kl_beta_implicit, kl_discrete, kl_concrete, child_stick_breaking_probs, global_stick_breaking_probs, reparameterize_discrete, implicit_beta

from IBP_BNN_multihead import IBP_BNN

"""MFVI BNN + H-IBP for CL"""
class HIBP_BNN(IBP_BNN):
    def __init__(self, alpha, input_size, hidden_size, output_size, training_size,
                 no_train_samples=10, no_pred_samples=100, num_ibp_samples=10, prev_means=None, prev_log_variances=None,
                 prev_betas=None, learning_rate=0.001, learning_rate_decay=0.87,
                 prior_mean=0, prior_var=1, alpha0=5., beta0=1., lambda_1=1., lambda_2=1.,
                 tensorboard_dir='logs', name='ibp', tb_logging=True, output_tb_gradients=False,
                 beta_1=1.0, beta_2=1.0, beta_3=1.0, use_local_reparam=True, implicit_beta=True,
                 clip_grads=False):

        super(HIBP_BNN, self).__init__(input_size=input_size, hidden_size=hidden_size,
                                       output_size=output_size, training_size=training_size,
                                       no_train_samples=no_train_samples, no_pred_samples=no_pred_samples,
                                       num_ibp_samples=num_ibp_samples, prev_means=prev_means,
                                       prev_log_variances=prev_log_variances, prev_betas=prev_betas,
                                       learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                                       prior_mean=prior_mean,
                                       prior_var=prior_var, alpha0=alpha0, beta0=beta0, lambda_1=lambda_1,
                                       lambda_2=lambda_2, tensorboard_dir=tensorboard_dir,
                                       name=name, tb_logging=tb_logging,
                                       output_tb_gradients=output_tb_gradients, beta_1=beta_1,
                                       beta_2=beta_2, beta_3=beta_3, use_local_reparam=use_local_reparam,
                                       implicit_beta=implicit_beta, clip_grads=clip_grads)
        self.alpha = alpha # hyper-param for each child IBP

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
                                 self.prior_mean, self.prior_var)
        self.prior_W_m, self.prior_b_m, self.prior_W_last_m, self.prior_b_last_m = m[0], m[1], m[2], m[3]
        self.prior_W_v, self.prior_b_v, self.prior_W_last_v, self.prior_b_last_v = v[0], v[1], v[2], v[3]

        betas = self.create_prior_betas(self.prev_betas, self.hidden_size)
        self.prior_gbeta_a, self.prior_gbeta_b = betas[0], betas[1]

        self.pred, prior_log_pis_bern, log_pis_bern, z_log_sample = self._prediction(self.x, self.task_idx, self.no_pred_samples)

        self.kl = tf.div(self._KL_term(self.num_ibp_samples, prior_log_pis_bern, log_pis_bern, z_log_sample),
                         self.training_size)
        self.ll = self._logpred(self.x, self.y, self.task_idx)
        self.cost = self.kl - self.ll
        self.acc = self._accuracy(self.x, self.y, self.task_idx)

        self.assign_optimizer(self.learning_rate)

        self.saver = tf.train.Saver()

        if self.tb_logging:
            self.create_summaries()

        self.assign_session()

    def assign_optimizer(self, learning_rate=0.001):
        #self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.compat.v1.train.exponential_decay(learning_rate,
                                                                  global_step,
                                                                  1000, self.learning_rate_decay, staircase=False)

        optim = tf.train.AdamOptimizer(self.learning_rate)
        gradients = optim.compute_gradients(self.cost)
        if self.tb_logging and self.output_tb_gradients:
            for grad_var_tuple in gradients:
                current_variable = grad_var_tuple[1]
                current_gradient = grad_var_tuple[0]
                if current_gradient is None:
                    # the subclass variables are also being created in the graph :(
                    # Hack, just skip these variables when getting the gradients
                    continue
                gradient_name_to_save = current_variable.name.replace(':', '_')  # tensorboard doesn't accept ':' symbol
                tf.compat.v1.summary.histogram(gradient_name_to_save, current_gradient)
        if self.clip_grads:
            grads, variables = zip(*gradients)
            _grads, _ = tf.clip_by_global_norm(grads, 10.0)
            gradients = zip(_grads, variables)
        self.train_step = optim.apply_gradients(grads_and_vars=gradients, global_step=global_step)

    # this samples a layer at a time
    def _prediction_layer(self, inputs, task_idx, no_samples):
        """ Outputs a prediction from the IBP BNN

        :param inputs: input tensor
        :param task_idx: tf placeholder
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
        self.global_log_pi = global_stick_breaking_probs(gbeta_a, gbeta_b,
                                                         size=(no_samples_ibp,), implicit=self.implicit_beta) # no_samples, dout
        self.prior_global_log_pi = global_stick_breaking_probs(prior_gbeta_a, prior_gbeta_b,
                                                               size=(no_samples_ibp,), implicit=self.implicit_beta) # no_samples, dout
        for i in range(self.no_layers - 1):
            alpha = self.alpha
            din = self.size[i]
            dout = self.size[i + 1]
            eps_w = tf.random_normal((no_samples, din, dout), 0, 1, dtype=tf.float32)
            eps_b = tf.random_normal((no_samples, 1, dout), 0, 1, dtype=tf.float32)

            # Gaussian re-parameterization
            _weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * self.W_v[i])), self.W_m[i]) # in [K, in, out]
            _biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * self.b_v[i])), self.b_m[i])

            # H-IBP
            prior_log_pi = child_stick_breaking_probs(tf.reduce_mean(tf.exp(self.prior_global_log_pi), 0),
                                                      alpha, size=(no_samples_ibp, batch_size, dout))
            prior_log_pis.append(prior_log_pi)
            self.log_pi = child_stick_breaking_probs(tf.reduce_mean(tf.exp(self.global_log_pi), 0),
                                                     alpha, size=(no_samples_ibp, batch_size, dout)) # (no_samples, batch_size, dout)
            log_pis.append(self.log_pi)
            # Concrete reparam
            z_log_sample = reparameterize_discrete(self.log_pi, self.lambda_1, size=(no_samples_ibp, batch_size, dout))
            z_discrete = tf.expand_dims(tf.reduce_mean(tf.sigmoid(z_log_sample), axis=0), 0)# (no_samples_ibp, batch_size, dout) --> (1, batch_size, dout)

            self.Z.append(z_discrete)
            self.vars += [tf.exp(0.5 * self.W_v[i]), tf.exp(0.5 * self.b_v[i])]
            self.means += [self.W_m[i], self.b_m[i]]
            log_z_sample.append(z_log_sample)
            # multiplication by IBP mask
            pre = tf.add(tf.einsum('mni,mio->mno', act, _weights), _biases) # m = samples, n = din, i=input d, o=output d
            act = tf.multiply(tf.nn.relu(pre), z_discrete) # [K, din, dout]

            # apply local reparameterisation trick: sample activations before applying the non-linearity
            m_h = tf.einsum('mni,io->mno', act_local, self.W_m[i])
            m_h = m_h + self.b_m[i]
            v_h = tf.einsum('mni,io->mno', tf.square(act_local), tf.exp(self.W_v[i])) # e^0.5 x ** 2 = e^x
            v_h = v_h + tf.exp(self.b_v[i])
            eps = tf.random_normal([no_samples, 1, dout], 0.0, 1.0, dtype=tf.float32)
            pre_local = m_h + tf.sqrt(v_h + 1e-9) * eps
            act_local = tf.multiply(tf.nn.relu(pre_local), z_discrete)  # shape (K, batch_size (None, ?), out)

        din = self.size[-2]
        dout = self.size[-1]
        eps_w = tf.random_normal((no_samples, din, dout), 0, 1, dtype=tf.float32)
        eps_b = tf.random_normal((no_samples, 1, dout), 0, 1, dtype=tf.float32)

        Wtask_m = tf.gather(self.W_last_m, task_idx)
        Wtask_v = tf.gather(self.W_last_v, task_idx)
        btask_m = tf.gather(self.b_last_m, task_idx)
        btask_v = tf.gather(self.b_last_v, task_idx)

        # Local reparam debug - can't use einsum :(
        # this is a sanity check
        # import numpy as np
        # >>> n=2
        # >>> I=3
        # >>> O=4
        # >>> b=5
        # >>> W = np.arange(I*O).reshape((I, O))
        # >>> A = np.arange(n*b*I).reshape((n, b, I))
        # >>> B = np.einsum('ijk, ko->ijo', W, A)
        # >>> B_bar = np.sum(np.multiply(A[:,:,:,np.newaxis], W[np.newaxis, np.newaxis, :, :]), 2)
        # B and B_bar are the same.

        _weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * Wtask_v)), Wtask_m)
        _biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * btask_v)), btask_m)
        self.vars += [tf.exp(0.5 * Wtask_v), tf.exp(0.5 * btask_v)]
        self.means += [Wtask_m, btask_m]

        act = tf.expand_dims(act, 3) # [no_samples, batch_size, din, 1]
        _weights = tf.expand_dims(_weights, 1) # [no_samples, 1, din, dout]
        pre = tf.add(tf.reduce_sum(act * _weights, 2), _biases)

        # apply local reparam trick to final output layer
        #m_h = tf.einsum('mni,io->mno', act_local, Wtask_m)
        self.m_h = tf.reduce_sum(tf.multiply(tf.expand_dims(act_local, 3),
                                        tf.expand_dims(tf.expand_dims(Wtask_m, 0), 0)), 2)
        m_h = self.m_h + btask_m
        #v_h = tf.einsum('mni,io->mno', tf.square(act_local), tf.exp(Wtask_v))
        self.v_h = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.square(act_local), 3),
                                        tf.expand_dims(tf.expand_dims(tf.exp(Wtask_v), 0), 0)), 2)
        v_h = self.v_h + tf.exp(btask_v)
        eps = tf.random_normal((no_samples, 1, dout), 0.0, 1.0, dtype=tf.float32)
        pre_local = self.m_h + tf.sqrt(self.v_h + 1e-9) * eps
        pre_local = tf.reshape(pre_local, [no_samples, batch_size, dout])
        return pre, pre_local, prior_log_pis, log_pis, log_z_sample

    def create_summaries(self):
        """Creates summaries in TensorBoard"""
        with tf.name_scope("summaries"):
            tf.compat.v1.summary.scalar("learning_rate", self.learning_rate)
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
            tf.compat.v1.summary.histogram("v_beta_a_l", tf.cast(tf.math.softplus(self.gbeta_a) + 0.01, tf.float32))
            tf.compat.v1.summary.histogram("v_beta_b_l", tf.cast(tf.math.softplus(self.gbeta_b) + 0.01, tf.float32))
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
            alpha = self.alpha
            # pis \in [np_ibp_samples, dout]
            kl_beta += kl_beta_implicit(alpha * tf.reduce_mean(tf.exp(self.global_log_pi), 0),
                                        alpha*(1-tf.reduce_mean(tf.exp(self.global_log_pi), 0)),
                                        alpha * tf.reduce_mean(tf.exp(self.prior_global_log_pi), 0),
                                        alpha*(1-tf.reduce_mean(tf.exp(self.prior_global_log_pi), 0)))
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
        # the initialisation of the Beta params should not be the same as the prior, as there is a risk that the KL
        # terms becomes close to zero, this seems to cause instabilities... Needs checking... TODO
        #
        # initial values are passed through log(exp(x)-1), this is the inverse of the softplus operation which these
        # are then passed to before sampling from the Beta distribution.
        hidden_size = deepcopy(hidden_size)
        dout = hidden_size[-1]
        if prev_betas is None:
            global_beta_a_val = tf.constant(np.log(np.exp(self.alpha0) - 1.), shape=[dout], dtype=tf.float32)
            global_beta_b_val = tf.constant(np.log(np.exp(self.beta0) - 1.), shape=[dout], dtype=tf.float32)
        else:
            global_beta_a_val = prev_betas[0]
            global_beta_b_val = prev_betas[1]
        # global_beta_a_val = tf.constant(np.log(np.exp(self.alpha0) - 1.), shape=[dout], dtype=tf.float32)
        # global_beta_b_val = tf.constant(np.log(np.exp(self.beta0) - 1.), shape=[dout], dtype=tf.float32)

        gb_a = tf.Variable(global_beta_a_val, name="global_beta_a", dtype=tf.float32)
        gb_b = tf.Variable(global_beta_b_val, name="global_beta_b", dtype=tf.float32)

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
        # Global beta param prior for H-IBP
        # For T_1, no prev_betas, the prior needs to be passed through the inverse of the softplus operation first
        hidden_size = deepcopy(hidden_size)
        dout = hidden_size[-1]
        if prev_betas is None:
            #global_beta_a_val = np.full((dout), self.alpha0)
            #global_beta_b_val = np.full((dout), self.beta0)
            global_beta_a_val = np.full((dout), np.log(np.exp(self.alpha0) - 1.0))
            global_beta_b_val = np.full((dout), np.log(np.exp(self.beta0) - 1.0))
        else:
            global_beta_a_val = prev_betas[0]
            global_beta_b_val = prev_betas[1]

        return [global_beta_a_val, global_beta_b_val]

    def create_prior(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prior_mean, prior_var):
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
            if prev_weights is not None and prev_variances is not None:
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

    def train_debug(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5, verbose=True):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        sess = self.sess
        costs = []
        global_step = 1

        writer = tf.compat.v1.summary.FileWriter(self.log_folder, sess.graph)
        # Training cycle
        for epoch in range(no_epochs):
            perm_inds = list(range(x_train.shape[0]))
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(total_batch):
                print("global step: {}".format(global_step))
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]

                # summary, v_alpha, v_beta, log_pis = sess.run([self.summary_op, self.gbeta_a, self.gbeta_b,
                #                                      self.global_log_pi],
                #                 feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx,
                #                            self.training: True})
                # writer.add_summary(summary, global_step)
                # TODO: add temp it not obsolete
                m, v = sess.run([self.m_h, self.v_h],
                                feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx,
                                           self.training: True})

                pdb.set_trace()
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run(
                    [self.train_step, self.cost],
                    feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx, self.training: True})

                # Compute average loss
                avg_cost += c / total_batch

                global_step += 1

            # Display logs per epoch step
            if verbose and epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "train cost=", \
                    "{:.9f}".format(avg_cost))
            costs.append(avg_cost)
        self.save(self.log_folder)
        writer.close()
        return costs

