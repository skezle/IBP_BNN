import tensorflow as tf
import numpy as np
from copy import deepcopy
from ddm.alg.utils import kl_beta_reparam, kl_discrete, kl_concrete, reparameterize_beta, reparameterize_discrete

np.random.seed(0)
tf.set_random_seed(0)

eps = 1e-16

# variable initialization functions
def weight_variable(shape, init_weights=None):
    if init_weights is not None:
        initial = tf.constant(init_weights)
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def small_variable(shape):
    initial = tf.constant(-6.0, shape=shape)
    return tf.Variable(initial)

def zero_variable(shape):
    initial = tf.zeros(shape=shape)
    return tf.Variable(initial)

def _create_weights_mf(in_dim, hidden_size, out_dim, init_weights=None, init_variances=None):
    size = deepcopy(hidden_size)
    size.append(out_dim)
    size.insert(0, in_dim)
    no_params = 0
    for i in range(len(size) - 1):
        no_weights = size[i] * size[i+1]
        no_biases = size[i+1]
        no_params += (no_weights + no_biases)
    m_weights = weight_variable([no_params], init_weights)
    if init_variances is None:
        v_weights = small_variable([no_params])
    else:
        v_weights = tf.Variable(tf.constant(init_variances, dtype=tf.float32))
    return no_params, m_weights, v_weights, size

class Cla_NN(object):
    def __init__(self, input_size, hidden_size, output_size, training_size):
        # input and output placeholders
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])
        self.task_idx = tf.placeholder(tf.int32)
        
    def assign_optimizer(self, learning_rate=0.001):
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def assign_session(self):
        # Initializing the variables
        init = tf.global_variables_initializer()

        # launch a session
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        sess = self.sess
        costs = []
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
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run(
                    [self.train_step, self.cost], 
                    feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            costs.append(avg_cost)
        print("Optimization Finished!")
        return costs

    def prediction(self, x_test, task_idx):
        # Test model
        prediction = self.sess.run([self.pred], feed_dict={self.x: x_test, self.task_idx: task_idx})[0]
        return prediction

    def prediction_prob(self, x_test, task_idx):
        prob = self.sess.run([tf.nn.softmax(self.pred)], feed_dict={self.x: x_test, self.task_idx: task_idx})[0]
        return prob

    def get_weights(self):
        weights = self.sess.run([self.weights])[0]
        return weights

    def close_session(self):
        self.sess.close()


""" Neural Network Model """
class Vanilla_NN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, prev_weights=None, learning_rate=0.001):

        super(Vanilla_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        # init weights and biases
        self.W, self.b, self.W_last, self.b_last, self.size = self.create_weights(
                input_size, hidden_size, output_size, prev_weights)
        self.no_layers = len(hidden_size) + 1
        self.pred = self._prediction(self.x, self.task_idx)
        self.cost = - self._logpred(self.x, self.y, self.task_idx)
        self.weights = [self.W, self.b, self.W_last, self.b_last]

        self.assign_optimizer(learning_rate)
        self.assign_session()

    def _prediction(self, inputs, task_idx):
        act = inputs
        for i in range(self.no_layers-1):
            pre = tf.add(tf.matmul(act, self.W[i]), self.b[i])
            act = tf.nn.relu(pre)
        pre = tf.add(tf.matmul(act, tf.gather(self.W_last, task_idx)), tf.gather(self.b_last, task_idx))
        return pre

    def _logpred(self, inputs, targets, task_idx):
        pred = self._prediction(inputs, task_idx)
        log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
        return log_lik

    def create_weights(self, in_dim, hidden_size, out_dim, prev_weights):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_params = 0
        no_layers = len(hidden_size) - 1
        W = []
        b = []
        W_last = []
        b_last = []
        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]
            if prev_weights is None:
                Wi_val = tf.truncated_normal([din, dout], stddev=0.1)
                bi_val = tf.truncated_normal([dout], stddev=0.1)
            else:
                Wi_val = tf.constant(prev_weights[0][i])
                bi_val = tf.constant(prev_weights[1][i])
            Wi = tf.Variable(Wi_val)
            bi = tf.Variable(bi_val)
            W.append(Wi)
            b.append(bi)

        if prev_weights is not None:
            prev_Wlast = prev_weights[2]
            prev_blast = prev_weights[3]
            no_prev_tasks = len(prev_Wlast)
            for j in range(no_prev_tasks):
                W_j = prev_Wlast[j]
                b_j = prev_blast[j]
                Wi = tf.Variable(W_j)
                bi = tf.Variable(b_j)
                W_last.append(Wi)
                b_last.append(bi)

        din = hidden_size[-2]
        dout = hidden_size[-1]
        Wi_val = tf.truncated_normal([din, dout], stddev=0.1)
        bi_val = tf.truncated_normal([dout], stddev=0.1)
        Wi = tf.Variable(Wi_val)
        bi = tf.Variable(bi_val)
        W_last.append(Wi)
        b_last.append(bi)
            
        return W, b, W_last, b_last, hidden_size


""" Bayesian Neural Network with Mean field VI approximation """
class MFVI_NN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size, 
        no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None, learning_rate=0.001, 
        prior_mean=0, prior_var=1):

        super(MFVI_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        m, v, self.size = self.create_weights(
            input_size, hidden_size, output_size, prev_means, prev_log_variances)
        self.W_m, self.b_m, self.W_last_m, self.b_last_m = m[0], m[1], m[2], m[3]
        self.W_v, self.b_v, self.W_last_v, self.b_last_v = v[0], v[1], v[2], v[3]
        self.weights = [m, v]

        m, v = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances, prior_mean, prior_var)
        self.prior_W_m, self.prior_b_m, self.prior_W_last_m, self.prior_b_last_m = m[0], m[1], m[2], m[3]
        self.prior_W_v, self.prior_b_v, self.prior_W_last_v, self.prior_b_last_v = v[0], v[1], v[2], v[3]

        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.pred = self._prediction(self.x, self.task_idx, self.no_pred_samples)
        self.cost = tf.div(self._KL_term(), training_size) - self._logpred(self.x, self.y, self.task_idx)
        
        self.assign_optimizer(learning_rate)
        self.assign_session()

    def _prediction(self, inputs, task_idx, no_samples):
        return self._prediction_layer(inputs, task_idx, no_samples)

    # this samples a layer at a time
    def _prediction_layer(self, inputs, task_idx, no_samples):
        K = no_samples
        act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])        
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
            eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)
            
            weights = tf.add(tf.multiply(eps_w, tf.exp(0.5*self.W_v[i])), self.W_m[i])
            biases = tf.add(tf.multiply(eps_b, tf.exp(0.5*self.b_v[i])), self.b_m[i])
            pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases)
            act = tf.nn.relu(pre)
        din = self.size[-2]
        dout = self.size[-1]
        eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
        eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)

        Wtask_m = tf.gather(self.W_last_m, task_idx)
        Wtask_v = tf.gather(self.W_last_v, task_idx)
        btask_m = tf.gather(self.b_last_m, task_idx)
        btask_v = tf.gather(self.b_last_v, task_idx)
        weights = tf.add(tf.multiply(eps_w, tf.exp(0.5*Wtask_v)), Wtask_m)
        biases = tf.add(tf.multiply(eps_b, tf.exp(0.5*btask_v)), btask_m)
        act = tf.expand_dims(act, 3)
        weights = tf.expand_dims(weights, 1)
        pre = tf.add(tf.reduce_sum(act * weights, 2), biases)

        return pre

    def _logpred(self, inputs, targets, task_idx):
        pred = self._prediction(inputs, task_idx, self.no_train_samples)
        targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
        log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
        return log_lik

    def _KL_term(self):
        kl = 0
        for i in range(self.no_layers-1):
            din = self.size[i]
            dout = self.size[i+1]
            m, v = self.W_m[i], self.W_v[i]
            m0, v0 = self.prior_W_m[i], self.prior_W_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_m[i], self.b_v[i]
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]
        for i in range(no_tasks):
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m)**2) / v0)
            kl += const_term + log_std_diff + mu_diff_term
        return kl

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
        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]
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

            Wi_m = tf.Variable(Wi_m_val)
            bi_m = tf.Variable(bi_m_val)
            Wi_v = tf.Variable(Wi_v_val)
            bi_v = tf.Variable(bi_v_val)
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
                Wi_m = tf.Variable(W_i_m)
                bi_m = tf.Variable(b_i_m)

                W_i_v = prev_Wlast_v[i]
                b_i_v = prev_blast_v[i]
                Wi_v = tf.Variable(W_i_v)
                bi_v = tf.Variable(b_i_v)
                
                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)

        din = hidden_size[-2]
        dout = hidden_size[-1]

        # if point estimate is supplied
        if prev_weights is not None and prev_variances is None:
            Wi_m_val = prev_weights[2][0]
            bi_m_val = prev_weights[3][0]
        else:
            Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
            bi_m_val = tf.truncated_normal([dout], stddev=0.1)
        Wi_v_val = tf.constant(-6.0, shape=[din, dout])
        bi_v_val = tf.constant(-6.0, shape=[dout])

        Wi_m = tf.Variable(Wi_m_val)
        bi_m = tf.Variable(bi_m_val)
        Wi_v = tf.Variable(Wi_v_val)
        bi_v = tf.Variable(bi_v_val)
        W_last_m.append(Wi_m)
        b_last_m.append(bi_m)
        W_last_v.append(Wi_v)
        b_last_v.append(bi_v)
            
        return [W_m, b_m, W_last_m, b_last_m], [W_v, b_v, W_last_v, b_last_v], hidden_size

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
        for i in range(no_layers-1):
            din = hidden_size[i]
            dout = hidden_size[i+1]
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


""" Bayesian Neural Network with Mean field VI approximation + IBP"""
class MFVI_IBP_NN(Cla_NN):
    def __init__(self, input_size, hidden_size, output_size, training_size,
                 no_train_samples=10, no_pred_samples=100, prev_means=None, prev_log_variances=None,
                 prev_betas_a=None, prev_betas_b=None, learning_rate=0.001,
                 prior_mean=0, prior_var=1, alpha0=5., temp=1., temp_prior=1., max_truncation_level=100.0):

        super(MFVI_IBP_NN, self).__init__(input_size, hidden_size, output_size, training_size)
        self.alpha0 = alpha0
        self.temp = temp
        self.temp_prior = temp_prior
        self.truncation = max_truncation_level
        self.training = tf.placeholder(tf.bool)
        m, v, betas, self.size = self.create_weights(
            input_size, hidden_size, output_size, prev_means, prev_log_variances, prev_betas_a, prev_betas_b)
        self.W_m, self.b_m, self.W_last_m, self.b_last_m = m[0], m[1], m[2], m[3]
        self.W_v, self.b_v, self.W_last_v, self.b_last_v = v[0], v[1], v[2], v[3]
        self.beta_a_W, self.beta_b_W, self.beta_a_b, self.beta_b_b, self.beta_last_a_W, self.beta_last_b_W, \
            self.beta_last_a_b, self.beta_last_b_b = betas[0], betas[1], betas[2], betas[3], betas[4], betas[5], betas[6], betas[7]

        self.weights = [m, v, betas]

        # used for the calculation of the KL term
        m, v, betas = self.create_prior(input_size, hidden_size, output_size, prev_means, prev_log_variances,
                                        prev_betas_a, prev_betas_b, prior_mean, prior_var)
        self.prior_W_m, self.prior_b_m, self.prior_W_last_m, self.prior_b_last_m = m[0], m[1], m[2], m[3]
        self.prior_W_v, self.prior_b_v, self.prior_W_last_v, self.prior_b_last_v = v[0], v[1], v[2], v[3]
        self.prior_beta_a_W, self.prior_beta_a_b, self.prior_beta_b_W, self.prior_beta_b_b, \
                self.prior_beta_last_a_W, self.prior_beta_last_a_b, self.prior_beta_last_b_W,\
                    self.prior_beta_last_b_b = betas[0], betas[1], betas[2], betas[3], betas[4], betas[5], betas[6], betas[7]

        self.no_layers = len(self.size) - 1
        self.no_train_samples = no_train_samples
        self.no_pred_samples = no_pred_samples
        self.pred = self._prediction(self.x, self.task_idx, self.no_pred_samples)
        self.kl = tf.div(self._KL_term(self.no_pred_samples), training_size)
        self.ll = self._logpred(self.x, self.y, self.task_idx)
        self.cost = self.kl - self.ll

        self.assign_optimizer(learning_rate)
        self.assign_session()

        self.create_summaries()

    def _prediction(self, inputs, task_idx, no_samples):
        return self._prediction_layer(inputs, task_idx, no_samples)

    # this samples a layer at a time
    def _prediction_layer(self, inputs, task_idx, no_samples):
        # inputs # [batch, d]
        K = no_samples
        act = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1]) # [1, batch, d] --> [K, batch, d]
        self.Z_W = []
        self.Z_b = []
        for i in range(self.no_layers - 1):
            din = self.size[i]
            dout = self.size[i + 1]
            eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
            eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)

            # Gaussian re-parameterization
            _weights = tf.add(tf.multiply(eps_w, tf.exp(0.5 * self.W_v[i])), self.W_m[i]) # in [K, in, out]
            _biases = tf.add(tf.multiply(eps_b, tf.exp(0.5 * self.b_v[i])), self.b_m[i])

            # IBP
            # beta reparam
            beta_a_W = tf.cast(tf.math.softplus(self.beta_a_W[i]) + 0.01, tf.float32) # log(1+e^x), beta_a \in [din, dout]
            beta_b_W = tf.cast(tf.math.softplus(self.beta_b_W[i]) + 0.01, tf.float32)
            beta_a_b = tf.cast(tf.math.softplus(self.beta_a_b[i]) + 0.01, tf.float32)
            beta_b_b = tf.cast(tf.math.softplus(self.beta_b_b[i]) + 0.01, tf.float32)
            logpis_W = reparameterize_beta(beta_a_W, beta_b_W, size=(K, din, dout), ibp=True, log=True)
            logpis_b = reparameterize_beta(beta_a_b, beta_b_b, size=(K, 1, dout), ibp=True, log=True)

            # Concrete reparam
            z_sample_W = reparameterize_discrete(logpis_W, self.temp, size=(K, din, dout))
            z_sample_b = reparameterize_discrete(logpis_b, self.temp, size=(K, 1, dout))
            self.Z_W.append(z_sample_W)
            self.Z_b.append(z_sample_b)
            z_discrete_W = tf.sigmoid(z_sample_W)
            z_discrete_b = tf.sigmoid(z_sample_b)

            # multiplication by IBP mask
            weights = tf.multiply(_weights, z_discrete_W)
            biases = tf.multiply(_biases, z_discrete_b)
            pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases) # m = samples, n = batch_size, i=input d, o=output d
            act = tf.nn.relu(pre)
        din = self.size[-2]
        dout = self.size[-1]
        eps_w = tf.random_normal((K, din, dout), 0, 1, dtype=tf.float32)
        eps_b = tf.random_normal((K, 1, dout), 0, 1, dtype=tf.float32)

        Wtask_m = tf.gather(self.W_last_m, task_idx)
        Wtask_v = tf.gather(self.W_last_v, task_idx)
        btask_m = tf.gather(self.b_last_m, task_idx)
        btask_v = tf.gather(self.b_last_v, task_idx)
        # IBP
        a_task_W = tf.gather(self.beta_last_a_W, task_idx)
        b_task_W = tf.gather(self.beta_last_b_W, task_idx)
        a_task_b = tf.gather(self.beta_last_a_b, task_idx)
        b_task_b = tf.gather(self.beta_last_b_b, task_idx)
        beta_a_W = tf.cast(tf.math.softplus(a_task_W) + 0.01, tf.float32)  # log(1+e^x), beta_a \in [din, dout]
        beta_b_W = tf.cast(tf.math.softplus(b_task_W) + 0.01, tf.float32)
        beta_a_b = tf.cast(tf.math.softplus(a_task_b) + 0.01, tf.float32)
        beta_b_b = tf.cast(tf.math.softplus(b_task_b) + 0.01, tf.float32)
        logpis_W = reparameterize_beta(beta_a_W, beta_b_W, size=(K, din, dout), ibp=True, log=True)
        logpis_b = reparameterize_beta(beta_a_b, beta_b_b, size=(K, 1, dout), ibp=True, log=True)

        # Concrete reparam
        z_sample_W = reparameterize_discrete(logpis_W, self.temp, size=(K, din, dout))
        z_sample_b = reparameterize_discrete(logpis_b, self.temp, size=(K, 1, dout))
        z_discrete_W = tf.sigmoid(z_sample_W)
        z_discrete_b = tf.sigmoid(z_sample_b)

        _weights = tf.multiply(tf.add(tf.multiply(eps_w, tf.exp(0.5 * Wtask_v)), Wtask_m), z_discrete_W)
        _biases = tf.multiply(tf.add(tf.multiply(eps_b, tf.exp(0.5 * btask_v)), btask_m), z_discrete_b)
        act = tf.expand_dims(act, 3)
        _weights = tf.expand_dims(_weights, 1)
        print("_weights: {}".format(_biases.get_shape()))
        print("_biases: {}".format(_biases.get_shape()))
        print("act: {}".format(act.get_shape()))
        pre = tf.add(tf.reduce_sum(act * _weights, 2), _biases)

        return pre

    def lof(self, x):
        # TODO
        return x

    def create_summaries(self):
        """Creates summaries in TensorBoard"""
        with tf.name_scope("summaries"):
            tf.summary.scalar("elbo", self.cost)
            tf.summary.scalar("loglik", self.ll)
            tf.summary.scalar("kl", self.kl)
            for i in range(len(self.Z_W)):
                tf.summary.image("W_Z_{}".format(i), self.lof(self.Z_W[i]))
                tf.summary.image("b_Z_{}".format(i), self.lof(self.Z_b[i]))

            self.summary_op = tf.summary.merge_all()

    def _logpred(self, inputs, targets, task_idx):
        pred = self._prediction(inputs, task_idx, self.no_train_samples)
        targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
        log_lik = - tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=targets))
        return log_lik

    def _KL_Bernoulli(self, a, b, p_a, p_b, K, din, dout):
        """sample from variational/real Bernoulli distribution.
        According to Maddison et. al. (2017) we sample from ExpConcrete during training
        and Bernoulli during testing.
        :param a: Beta a param
        :param b: beta b param
        :param p_a: prior Beta a param
        :param p_b: prior Beta b param
        :param K: number of samples to make
        :param din: input dim for layer
        :param dout: output dim for layer
        :return:
        """
        beta_a_W = tf.cast(tf.math.softplus(a) + 0.01, tf.float32)  # log(1+e^x), beta_a \in [din, dout]
        beta_b_W = tf.cast(tf.math.softplus(b) + 0.01, tf.float32)
        prior_beta_a_W = tf.cast(tf.math.softplus(p_a) + 0.01, tf.float32)
        prior_beta_b_W = tf.cast(tf.math.softplus(p_b) + 0.01, tf.float32)
        logpis_W = reparameterize_beta(beta_a_W, beta_b_W, size=(K, din, dout), ibp=True, log=True)
        logpis_W_prior = reparameterize_beta(prior_beta_a_W, prior_beta_b_W, size=(K, din, dout), ibp=True, log=True)
        # Concrete reparam
        z_sample_W = reparameterize_discrete(logpis_W, self.temp, size=(K, din, dout))

        kl = tf.cond(self.training,
                     true_fn=lambda: kl_concrete(logpis_W, logpis_W_prior, z_sample_W, self.temp, self.temp_prior),
                     false_fn=lambda: kl_discrete(logpis_W, logpis_W_prior, z_sample_W))
        print("bern kl: {}".format(kl.get_shape()))
        return kl


    def _KL_term(self, no_samples):
        K = no_samples
        kl = 0
        for i in range(self.no_layers - 1):
            din = self.size[i]
            dout = self.size[i + 1]
            m, v = self.W_m[i], self.W_v[i]
            m0, v0 = self.prior_W_m[i], self.prior_W_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            kl += kl_beta_reparam(self.beta_a_W[i], self.beta_b_W[i],
                                  self.prior_beta_a_W[i], self.prior_beta_b_W[i])

            kl += self._KL_Bernoulli(self.beta_a_W[i], self.beta_b_W[i], self.prior_beta_a_W[i], self.prior_beta_b_W[i],
                                     K, din, dout)

            m, v = self.b_m[i], self.b_v[i]
            m0, v0 = self.prior_b_m[i], self.prior_b_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            kl += kl_beta_reparam(self.beta_a_b[i], self.beta_b_b[i], self.prior_beta_a_b[i], self.prior_beta_b_b[i])
            kl += self._KL_Bernoulli(self.beta_a_b[i], self.beta_b_b[i], self.prior_beta_a_b[i], self.prior_beta_b_b[i],
                                     K, din, dout)

        no_tasks = len(self.W_last_m)
        din = self.size[-2]
        dout = self.size[-1]
        for i in range(no_tasks):
            m, v = self.W_last_m[i], self.W_last_v[i]
            m0, v0 = self.prior_W_last_m[i], self.prior_W_last_v[i]
            const_term = -0.5 * dout * din
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            kl += kl_beta_reparam(self.beta_last_a_W[i], self.beta_last_b_W[i], self.prior_beta_last_a_W[i], self.prior_beta_last_b_W[i])

            kl += self._KL_Bernoulli(self.beta_last_a_W[i], self.beta_last_b_W[i], self.prior_beta_last_a_W[i],
                                     self.prior_beta_last_b_W[i], K, din, dout)

            m, v = self.b_last_m[i], self.b_last_v[i]
            m0, v0 = self.prior_b_last_m[i], self.prior_b_last_v[i]
            const_term = -0.5 * dout
            log_std_diff = 0.5 * tf.reduce_sum(np.log(v0) - v)
            mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v) + (m0 - m) ** 2) / v0)
            kl += const_term + log_std_diff + mu_diff_term

            kl += kl_beta_reparam(self.beta_last_a_b[i], self.beta_last_b_b[i], self.prior_beta_last_a_b[i], self.prior_beta_last_b_b[i])
            kl += self._KL_Bernoulli(self.beta_last_a_b[i], self.beta_last_b_b[i], self.prior_beta_last_a_b[i],
                                     self.prior_beta_last_b_b[i], K, din, dout)

        return kl

    def create_weights(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prev_betas_a, prev_betas_b):
        hidden_size = deepcopy(hidden_size)
        hidden_size.append(out_dim)
        hidden_size.insert(0, in_dim)
        no_params = 0
        no_layers = len(hidden_size) - 1
        # IBP prior
        W_m = []
        b_m = []
        W_last_m = []
        b_last_m = []
        W_v = []
        b_v = []
        W_last_v = []
        b_last_v = []
        W_b_a = []
        W_b_b = []
        b_b_a = []
        b_b_b = []
        W_beta_last_a = []
        W_beta_last_b = []
        b_beta_last_a = []
        b_beta_last_b = []
        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]
            if prev_weights is None:
                Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
                bi_m_val = tf.truncated_normal([dout], stddev=0.1)
                Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                bi_v_val = tf.constant(-6.0, shape=[dout])
                W_beta_a_val = tf.constant(np.log(np.exp(self.alpha0) - 1), shape=[din, dout])
                W_beta_b_val = tf.constant(np.log(np.exp(1.) - 1), shape=[din, dout])
                b_beta_a_val = tf.constant(np.log(np.exp(self.alpha0) - 1), shape=[dout])
                b_beta_b_val = tf.constant(np.log(np.exp(1.) - 1), shape=[dout])
            else:
                Wi_m_val = prev_weights[0][i]
                bi_m_val = prev_weights[1][i]
                W_beta_a_val = prev_betas_a[0][i]
                W_beta_b_val = prev_betas_b[0][i]
                b_beta_a_val = prev_betas_a[1][i]
                b_beta_b_val = prev_betas_b[1][i]
                if prev_variances is None:
                    Wi_v_val = tf.constant(-6.0, shape=[din, dout])
                    bi_v_val = tf.constant(-6.0, shape=[dout])
                else:
                    Wi_v_val = prev_variances[0][i]
                    bi_v_val = prev_variances[1][i]

            Wi_m = tf.Variable(Wi_m_val)
            bi_m = tf.Variable(bi_m_val)
            Wi_v = tf.Variable(Wi_v_val)
            bi_v = tf.Variable(bi_v_val)
            W_beta_a = tf.Variable(W_beta_a_val)
            W_beta_b = tf.Variable(W_beta_b_val)
            b_beta_a = tf.Variable(b_beta_a_val)
            b_beta_b = tf.Variable(b_beta_b_val)

            W_m.append(Wi_m)
            b_m.append(bi_m)
            W_v.append(Wi_v)
            b_v.append(bi_v)
            W_b_a.append(W_beta_a)
            W_b_b.append(W_beta_b)
            b_b_a.append(b_beta_a)
            b_b_b.append(b_beta_b)

        # if there are previous tasks
        if prev_weights is not None and prev_variances is not None and prev_betas_a is not None and prev_betas_b is not None:
            prev_Wlast_m = prev_weights[2]
            prev_blast_m = prev_weights[3]
            prev_Wlast_v = prev_variances[2]
            prev_blast_v = prev_variances[3]
            prev_W_beta_a = prev_betas_a[2] # TODO: indexing seems arbitrary.
            prev_b_beta_a = prev_betas_a[3]
            prev_W_beta_b = prev_betas_b[2]
            prev_b_beta_b = prev_betas_b[3]
            no_prev_tasks = len(prev_Wlast_m)
            for i in range(no_prev_tasks):
                W_i_m = prev_Wlast_m[i]
                b_i_m = prev_blast_m[i]
                Wi_m = tf.Variable(W_i_m)
                bi_m = tf.Variable(b_i_m)

                W_i_v = prev_Wlast_v[i]
                b_i_v = prev_blast_v[i]
                Wi_v = tf.Variable(W_i_v)
                bi_v = tf.Variable(b_i_v)

                W_beta_i_a_v = prev_W_beta_a[i]
                b_beta_i_a_v = prev_b_beta_a[i]
                W_beta_i_b_v = prev_W_beta_b[i]
                b_beta_i_b_v = prev_b_beta_b[i]
                W_beta_a = tf.Variable(W_beta_i_a_v)
                b_beta_a = tf.Variable(b_beta_i_a_v)
                W_beta_b = tf.Variable(W_beta_i_b_v)
                b_beta_b = tf.Variable(b_beta_i_b_v)
                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)
                W_beta_last_a.append(W_beta_a)
                W_beta_last_b.append(W_beta_b)
                b_beta_last_a.append(b_beta_a)
                b_beta_last_b.append(b_beta_b)

        din = hidden_size[-2]
        dout = hidden_size[-1]

        # if point estimate is supplied
        # no IBP params
        if prev_weights is not None and prev_variances is None:
            Wi_m_val = prev_weights[2][0]
            bi_m_val = prev_weights[3][0]
        else:
            Wi_m_val = tf.truncated_normal([din, dout], stddev=0.1)
            bi_m_val = tf.truncated_normal([dout], stddev=0.1)
        Wi_v_val = tf.constant(-6.0, shape=[din, dout])
        bi_v_val = tf.constant(-6.0, shape=[dout])
        W_beta_a_val = tf.constant(np.log(np.exp(self.alpha0) - 1), shape=[din, dout])
        W_beta_b_val = tf.constant(np.log(np.exp(1.) - 1), shape=[din, dout])
        b_beta_a_val = tf.constant(np.log(np.exp(self.alpha0) - 1), shape=[dout])
        b_beta_b_val = tf.constant(np.log(np.exp(1.) - 1), shape=[dout])

        Wi_m = tf.Variable(Wi_m_val)
        bi_m = tf.Variable(bi_m_val)
        Wi_v = tf.Variable(Wi_v_val)
        bi_v = tf.Variable(bi_v_val)
        _W_beta_a = tf.Variable(W_beta_a_val)
        _W_beta_b = tf.Variable(W_beta_b_val)
        _b_beta_a = tf.Variable(b_beta_a_val)
        _b_beta_b = tf.Variable(b_beta_b_val)
        W_last_m.append(Wi_m)
        b_last_m.append(bi_m)
        W_last_v.append(Wi_v)
        b_last_v.append(bi_v)
        W_beta_last_a.append(_W_beta_a)
        W_beta_last_b.append(_W_beta_b)
        b_beta_last_a.append(_b_beta_a)
        b_beta_last_b.append(_b_beta_b)

        return [W_m, b_m, W_last_m, b_last_m], [W_v, b_v, W_last_v, b_last_v], \
               [W_b_a, W_b_b, b_b_a, b_b_b, W_beta_last_a, W_beta_last_b, b_beta_last_a, b_beta_last_b], hidden_size

    def create_prior(self, in_dim, hidden_size, out_dim, prev_weights, prev_variances, prev_betas_a, prev_betas_b, prior_mean, prior_var):
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
        W_betas_a = []
        W_betas_b = []
        b_betas_a = []
        b_betas_b = []
        W_last_betas_a = []
        W_last_betas_b = []
        b_last_betas_a = []
        b_last_betas_b = []
        for i in range(no_layers - 1):
            din = hidden_size[i]
            dout = hidden_size[i + 1]
            if prev_weights is not None and prev_variances is not None and prev_betas_a is not None and prev_betas_b is not None:
                Wi_m = prev_weights[0][i]
                bi_m = prev_weights[1][i]
                Wi_v = np.exp(prev_variances[0][i])
                bi_v = np.exp(prev_variances[1][i])
                W_beta_a_v = prev_betas_a[0][i]
                W_beta_b_v = prev_betas_b[0][i]
                b_beta_a_v = prev_betas_a[0][i]
                b_beta_b_v = prev_betas_b[1][i]
            else:
                Wi_m = prior_mean
                bi_m = prior_mean
                Wi_v = prior_var
                bi_v = prior_var
                W_beta_a_v = self.alpha0
                W_beta_b_v = 1.0
                b_beta_a_v = self.alpha0
                b_beta_b_v = 1.0


            W_m.append(Wi_m)
            b_m.append(bi_m)
            W_v.append(Wi_v)
            b_v.append(bi_v)
            W_betas_a.append(W_beta_a_v)
            W_betas_b.append(W_beta_b_v)
            b_betas_a.append(b_beta_a_v)
            b_betas_b.append(b_beta_b_v)

        # if there are previous tasks
        if prev_weights is not None and prev_variances is not None and prev_betas_a is not None and prev_betas_b is not None:
            prev_Wlast_m = prev_weights[2]
            prev_blast_m = prev_weights[3]
            prev_Wlast_v = prev_variances[2]
            prev_blast_v = prev_variances[3]
            prev_W_beta_a = prev_betas_a[2] # TODO: check indexing
            prev_W_beta_b = prev_betas_b[2]
            prev_b_beta_a = prev_betas_a[3]
            prev_b_beta_b = prev_betas_b[3]
            no_prev_tasks = len(prev_Wlast_m)
            for i in range(no_prev_tasks):
                Wi_m = prev_Wlast_m[i]
                bi_m = prev_blast_m[i]
                Wi_v = np.exp(prev_Wlast_v[i])
                bi_v = np.exp(prev_blast_v[i])
                W_beta_i_a = prev_W_beta_a[i]
                W_beta_i_b = prev_W_beta_b[i]
                b_beta_i_a = prev_b_beta_a[i]
                b_beta_i_b = prev_b_beta_b[i]
                W_last_m.append(Wi_m)
                b_last_m.append(bi_m)
                W_last_v.append(Wi_v)
                b_last_v.append(bi_v)
                W_last_betas_a.append(W_beta_i_a)
                W_last_betas_b.append(W_beta_i_b)
                b_last_betas_a.append(b_beta_i_a)
                b_last_betas_b.append(b_beta_i_b)

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
        W_last_betas_a.append(self.alpha0)
        W_last_betas_b.append(1.0)
        b_last_betas_a.append(self.alpha0)
        b_last_betas_b.append(1.0)

        return [W_m, b_m, W_last_m, b_last_m], [W_v, b_v, W_last_v, b_last_v], \
               [W_betas_a, W_betas_b, b_betas_a, b_betas_b, W_last_betas_a, W_last_betas_b, b_last_betas_a, b_last_betas_b]

    def train(self, x_train, y_train, task_idx, no_epochs=1000, batch_size=100, display_epoch=5):
        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N

        sess = self.sess
        costs = []
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
                start_ind = i*batch_size
                end_ind = np.min([(i+1)*batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run(
                    [self.train_step, self.cost],
                    feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx, self.training: True})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_epoch == 0:
                _ = sess.run(self.summary_op,
                             feed_dict={self.x: batch_x, self.y: batch_y, self.task_idx: task_idx, self.training: True})
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            costs.append(avg_cost)
        print("Optimization Finished!")
        return costs

    def prediction(self, x_test, task_idx):
        # Test model
        prediction = self.sess.run([self.pred], feed_dict={self.x: x_test, self.task_idx: task_idx, self.training: False})[0]
        return prediction

    def prediction_prob(self, x_test, task_idx):
        prob = self.sess.run([tf.nn.softmax(self.pred)], feed_dict={self.x: x_test, self.task_idx: task_idx, self.training: False})[0]
        return prob
