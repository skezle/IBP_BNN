import numpy as np
import tensorflow as tf
from utils import get_scores, concatenate_results
from cla_models_multihead import Vanilla_NN, MFVI_NN, MFVI_IBP_NN

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, val=False,
            verbose=True, name='vcl'):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):

        tf.reset_default_graph()

        if val:
            x_train, y_train, x_test, y_test, _, _ = data_gen.next_task()
        else:
            x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            ml_model.close_session()

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Train on non-coreset data
        mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances,
                           name="{0}_task{1}".format(name, task_id+1))
        mf_model.train(x_train, y_train, head, no_epochs, bsize, display_epoch=5, verbose=verbose)
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        acc = get_scores(mf_model, x_testsets, y_testsets, single_head)
        all_acc = concatenate_results(acc, all_acc)

        mf_model.close_session()

    return all_acc

def run_vcl_ibp(hidden_size, no_epochs, data_gen, run_index, tag, dataset,
                val, batch_size=None, single_head=True, alpha0=5.0,
                beta0 = 1.0, lambda_1 = 1.0, lambda_2 = 1.0, learning_rate=0.0001,
                no_pred_samples=100, ibp_samples = 10):

    in_dim, out_dim = data_gen.get_dims()
    x_testsets, y_testsets = [], []
    Zs = []
    for task_id in range(data_gen.max_iter):

        tf.reset_default_graph()
        if val:
            x_train, y_train, x_test, y_test, _, _ = data_gen.next_task()
        else:
            x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        # lambda_1 --> temp of the variational Concrete posterior
        # lambda_2 --> temp of the relaxed prior, for task != 0 this should be lambda_1!!!
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            mf_betas = None
            ml_model.close_session()

        # Train on non-coreset data
        mf_model = MFVI_IBP_NN(in_dim, hidden_size, out_dim, x_train.shape[0], num_ibp_samples=ibp_samples,
                               prev_means=mf_weights,
                               prev_log_variances=mf_variances, prev_betas=mf_betas,
                               alpha0=alpha0, beta0=beta0, learning_rate=learning_rate,
                               lambda_1=lambda_1,
                               lambda_2=lambda_2 if task_id == 0 else lambda_1,
                               no_pred_samples=no_pred_samples,
                               name='ibp_{0}_run{1}_{2}_task{3}'.format(dataset, run_index + 1, tag,
                                                                        task_id + 1))

        mf_model.train(x_train, y_train, head, no_epochs, bsize,
                       anneal_rate=0.0, min_temp=1.0)
        mf_weights, mf_variances, mf_betas = mf_model.get_weights()

        acc = get_scores(mf_model, x_testsets, y_testsets, single_head)

        Zs.append(mf_model.sess.run(mf_model.Z, feed_dict={mf_model.x: x_test,
                                                           mf_model.task_idx: task_id,
                                                           mf_model.training: False, mf_model.temp: 1.0})[0])

        mf_model.close_session()

    return acc, Zs