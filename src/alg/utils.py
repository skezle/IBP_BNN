import numpy as np
import pdb
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from cla_models_multihead import MFVI_NN
from IBP_BNN_multihead import IBP_BNN
from HIBP_BNN_multihead import HIBP_BNN

eps = 1e-10

def get_uncertainties(model, x_testsets, y_testsets, single_head, task_id, bsize):
    # uncertainties of test set like in Uncertainty in Deep Learning
    # Gal p. 53.
    uncert = []
    for i in range(len(x_testsets)):
        head = 0 if single_head else (task_id if i > task_id else i) # ensures we use the final multi-head model which is available
        x_test, ytest = x_testsets[i], y_testsets[i]
        #mi = mutual_information(model, x_test, head)
        pe = predictive_entropy(model, x_test, head, bsize)
        uncert.append(pe)
    return uncert

def predictive_entropy(model, x_test, task_idx, bsize, stamp, reps=1):
    # pdb.set_trace()
    if stamp is not None:
        mc_samples = [model.prediction_prob(x_test, task_idx, bsize, stamp) for _ in range(reps)] # each item is a list of [no_preds, None, n_out]
    else:
        mc_samples = [model.prediction_prob(x_test, task_idx, bsize) for _ in range(reps)]  # each item is a list of [no_preds, None, n_out]
    mc_samples_ = [np.concatenate(item, axis=1) for item in mc_samples] # each item is a list of [no_preds, N_test, n_out]
    mc_samples__ = np.concatenate(mc_samples_, axis=0) # ar of size 10xno_preds, N_test, n_out
    # pdb.set_trace()
    mc_arr = mc_samples__.reshape(-1, mc_samples__.shape[-1]) # 10xno_predxtest_size, n_out
    predictive_entropy = -np.sum(mc_arr * np.log(mc_arr + eps), axis=-1)  # (test_set_size, )
    return predictive_entropy # test_size: (n,)

def mutual_information(model, x_test, task_idx, bsize, stamp, reps=1):
    """
    :param model: BNN model object
    :param x_test: test data (n, d)
    :param task_idx: int
    :return:
    """
    # expected entropy part
    pe = predictive_entropy(model, x_test, task_idx, bsize) # (test_size,)
    # expected entropy part
    if stamp is not None:
        mc_samples = [model.prediction_prob(x_test, task_idx, bsize, stamp) for _ in range(reps)]
    else:
        mc_samples = [model.prediction_prob(x_test, task_idx, bsize) for _ in range(reps)]
    mc_samples_ = [np.concatenate(item, axis=1) for item in mc_samples]  # each item is a list of [no_preds, N_test, n_out]
    mc_samples__ = np.concatenate(mc_samples_, axis=0)  # ar of size 10xno_preds, N_test, n_out
    mc_arr = mc_samples__.reshape(-1, mc_samples__.shape[-1])  # 10xno_predxtest_size, n_out
    mc_entropy = np.sum(mc_arr * np.log(mc_arr+eps), axis=-1) # (test_set_size, )
    expected_entropy = -np.mean(mc_entropy, axis=0) # (1, )
    mi = pe - expected_entropy
    return mi

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

def get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, batch_size, single_head, stamp,
               hparams, ibp, hibp):
    accs = []
    if ibp or hibp:
        mf_weights, mf_variances, betas = model.get_weights()
    else:
        mf_weights, mf_variances = model.get_weights()

    if len(x_coresets) > 0:

        if single_head:
            raise ValueError
        else:
            for i in range(len(x_coresets)):

                tf.compat.v1.reset_default_graph()

                x_train, y_train = x_coresets[i], y_coresets[i]
                if ibp:
                    final_model = IBP_BNN(x_train.shape[1], hparams.hidden_size, y_train.shape[1], x_train.shape[0],
                                    num_ibp_samples=hparams.ibp_samples, prev_means=mf_weights,
                                    prev_log_variances=mf_variances, prev_betas=betas, alpha0=hparams.alpha0,
                                    beta0=hparams.beta0, learning_rate=hparams.lr, prior_mean=hparams.prior_mean,
                                    prior_var=hparams.prior_var, lambda_1=hparams.lambda_1,
                                    lambda_2=hparams.lambda_2 if i == 0 else hparams.lambda_1,
                                    no_pred_samples=hparams.no_pred_samples,
                                    tensorboard_dir=hparams.log_dir, tb_logging=hparams.tb_logging,
                                    name='{0}_coreset{1}'.format(hparams.name, i + 1),
                                    use_local_reparam=hparams.use_local_reparam,
                                    implicit_beta=hparams.implicit_beta,
                                    beta_1=hparams.b1, beta_2=hparams.beta_2, beta_3=hparams.beta_3)
                    final_model.create_model()
                elif hibp:
                    final_model = HIBP_BNN(hparams.a, x_train.shape[1], hparams.hidden_size, y_train.shape[1],
                                     x_train.shape[0], num_ibp_samples=hparams.ibp_samples, prev_means=mf_weights,
                                     prev_log_variances=mf_variances, prev_betas=betas,
                                     alpha0=hparams.alpha0, beta0=hparams.beta0, learning_rate=hparams.lr,
                                     learning_rate_decay=hparams.learning_rate_decay,
                                     prior_mean=hparams.prior_mean, prior_var=hparams.prior_var, lambda_1=hparams.lambda_1,
                                     lambda_2=hparams.lambda_2 if i == 0 else hparams.lambda_1,
                                     no_pred_samples=hparams.no_pred_samples,
                                     tensorboard_dir=hparams.log_dir,
                                     tb_logging=hparams.tb_logging, name='{0}_coreset{1}'.format(hparams.name, i + 1),
                                     use_local_reparam=hparams.use_local_reparam, implicit_beta=hparams.implicit_beta,
                                     beta_1=hparams.b1, beta_2=hparams.beta_2, beta_3=hparams.beta_3)
                    final_model.create_model()
                else:
                    final_model = MFVI_NN(x_train.shape[1], hparams.hidden_size, y_train.shape[1], x_train.shape[0],
                                 prev_means=mf_weights, prev_log_variances=mf_variances,
                                 name="{0}_coreset{1}".format(hparams.name, i+1), tensorboard_dir=hparams.log_dir,
                                 use_local_reparam=hparams.use_local_reparam)
                print("Training model with coreset: {0}".format(i))
                final_model.train(x_train, y_train, task_idx=int(i), no_epochs=2000, batch_size=100)

                if ibp or hibp:
                    mf_weights, mf_variances, betas = final_model.get_weights()
                else:
                    mf_weights, mf_variances = final_model.get_weights()

                if len(x_coresets) - 1 > i:
                    print("Closing session")
                    final_model.close_session()
    else:
        final_model = model

    for i in range(len(x_testsets)):
        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        if stamp is not None:
            acc, _ = final_model.prediction_acc(x_test, y_test, batch_size, head, stamp[head+1])
        else:
            acc, _ = final_model.prediction_acc(x_test, y_test, batch_size, head)
        accs.append(acc)

    if len(x_coresets) > 0:
        final_model.close_session()

    return accs

def get_scores_entropy(model, x_testsets, y_testsets, x_coresets, y_coresets, single_head, stamp,
                       hparams, ibp, hibp, batch_size, pred_ent=True, use_uncert=True):
    accs, uncerts = [], []
    if ibp or hibp:
        mf_weights, mf_variances, betas = model.get_weights()
    else:
        mf_weights, mf_variances = model.get_weights()

    if len(x_coresets) > 0:

        if single_head:
            raise ValueError
        else:
            for i in range(len(x_coresets)):

                tf.compat.v1.reset_default_graph()

                x_train, y_train = x_coresets[i], y_coresets[i]
                if ibp:
                    final_model = IBP_BNN(x_train.shape[1], hparams.hidden_size, y_train.shape[1], x_train.shape[0],
                                    num_ibp_samples=hparams.ibp_samples, prev_means=mf_weights,
                                    prev_log_variances=mf_variances, prev_betas=betas, alpha0=hparams.alpha0,
                                    beta0=hparams.beta0, learning_rate=hparams.lr, prior_mean=hparams.prior_mean,
                                    prior_var=hparams.prior_var, lambda_1=hparams.lambda_1,
                                    lambda_2=hparams.lambda_2 if i == 0 else hparams.lambda_1,
                                    no_pred_samples=hparams.no_pred_samples,
                                    tensorboard_dir=hparams.log_dir, tb_logging=hparams.tb_logging,
                                    name='{0}_coreset_pe{1}'.format(hparams.name, i + 1),
                                    use_local_reparam=hparams.use_local_reparam,
                                    implicit_beta=hparams.implicit_beta,
                                    beta_1=hparams.b1, beta_2=hparams.beta_2, beta_3=hparams.beta_3)

                    final_model.create_model()
                elif hibp:
                    final_model = HIBP_BNN(hparams.a, x_train.shape[1], hparams.hidden_size, y_train.shape[1],
                                     x_train.shape[0], num_ibp_samples=hparams.ibp_samples, prev_means=mf_weights,
                                     prev_log_variances=mf_variances, prev_betas=betas,
                                     alpha0=hparams.alpha0, beta0=hparams.beta0, learning_rate=hparams.lr,
                                     learning_rate_decay=hparams.learning_rate_decay,
                                     prior_mean=hparams.prior_mean, prior_var=hparams.prior_var, lambda_1=hparams.lambda_1,
                                     lambda_2=hparams.lambda_2 if i == 0 else hparams.lambda_1,
                                     no_pred_samples=hparams.no_pred_samples,
                                     tensorboard_dir=hparams.log_dir,
                                     tb_logging=hparams.tb_logging, name='{0}_coreset_pe{1}'.format(hparams.name, i + 1),
                                     use_local_reparam=hparams.use_local_reparam, implicit_beta=hparams.implicit_beta,
                                     beta_1=hparams.b1, beta_2=hparams.beta_2, beta_3=hparams.beta_3)

                    final_model.create_model()
                else:
                    final_model = MFVI_NN(x_train.shape[1], hparams.hidden_size, y_train.shape[1], x_train.shape[0],
                                 prev_means=mf_weights, prev_log_variances=mf_variances,
                                 name="{0}_coreset_pe{1}".format(hparams.name, i+1), tensorboard_dir=hparams.log_dir,
                                 use_local_reparam=hparams.use_local_reparam)

                print("Training model with coreset: {0}".format(i))
                final_model.train(x_train, y_train, task_idx=int(i), no_epochs=2000, batch_size=100)

                if ibp or hibp:
                    mf_weights, mf_variances, betas = final_model.get_weights()
                else:
                    mf_weights, mf_variances = final_model.get_weights()

                if len(x_coresets) - 1 > i:
                    print("Closing session")
                    final_model.close_session()

    else:
        final_model = model

    for i in range(len(x_testsets)): # iterating over the test datasets
        uncerts_task, accs_task = [], []
        x_test, y_test = x_testsets[i], y_testsets[i]
        N = x_test.shape[0]
        b = 128 # batch size to pass to graph to make things efficient, passing the entire test set can lead to OOM issues.
        bsize = N if (batch_size is None) else batch_size # size of the dataset to aggregate over for the uncertainty measure
        alpha = 1 if use_uncert else 0
        num_batches = int(np.ceil(N * 1.0 / bsize))
        for k in range(num_batches):
            start_ind = k * bsize
            end_ind = np.min([(k + 1) * bsize, N])
            x_test_batch = x_test[start_ind:end_ind, :]
            y_test_batch = y_test[start_ind:end_ind, :]
            for j in range(len(x_testsets)): # iterating over the heads seen so far
                if stamp is not None:
                    s = stamp[j+1]
                else:
                    s = None
                if pred_ent:
                    u = predictive_entropy(final_model, x_test_batch, j, b, s) # differnet batch size
                else:
                    u = mutual_information(final_model, x_test_batch, j, b, s)
                obj = np.mean(u) - alpha*np.std(u) # Optimism,  Sharpe doesn't work
                uncerts_task.append(obj)
                uncerts.append((np.mean(u), np.std(u)))
            head = np.argmin(uncerts_task)
            if stamp is not None:
                acc, _ = final_model.prediction_acc(x_test_batch, y_test_batch, b, head, stamp[head+1])
            else:
                acc, _ = final_model.prediction_acc(x_test_batch, y_test_batch, b, head)
            accs_task.append(acc)
        accs.append(np.mean(accs_task))

    if len(x_coresets) > 0:
        final_model.close_session()

    return accs, uncerts

def get_Zs(model, x_test, batch_size, task_id):
    Zs = model.prediction_Zs(x_test, batch_size, task_id)
    return Zs
