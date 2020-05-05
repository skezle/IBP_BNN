import numpy as np
import pdb
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

eps = 1e-16

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

def get_scores_entropy(model, x_testsets, y_testsets, batch_size, num_tasks):
    accs = []
    for i in range(len(x_testsets)):
        uncerts, accs_task = [], []
        x_test, y_test = x_testsets[i], y_testsets[i]
        N = x_test.shape[0]
        num_batches = N / batch_size
        for k in range(num_batches):
            start_ind = k * batch_size
            end_ind = np.min([(k + 1) * batch_size, N])
            x_test_batch = x_test[start_ind:end_ind, :]
            y_test_batch = y_test[start_ind:end_ind, :]
            for j in range(num_tasks):
                pe = predictive_entropy(model, x_test_batch, j, batch_size)
                uncerts.append(-np.mean(pe) / np.std(pe)) # negative Sharpe
            head = np.argmax(uncerts)
            acc, _ = model.prediction_acc(x_test_batch, y_test_batch, batch_size, head)
            accs_task.append(acc)
        accs.append(np.mean(accs_task))
    return accs

def get_Zs(model, x_test, batch_size, task_id):
    Zs = model.prediction_Zs(x_test, batch_size, task_id)
    return Zs
