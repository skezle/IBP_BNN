import pdb
import os.path
import numpy as np
import tensorflow as tf
from utils import get_scores, get_scores_entropy, get_uncertainties, concatenate_results, get_Zs
from cla_models_multihead import Vanilla_NN, MFVI_NN
from IBP_BNN_multihead import IBP_BNN
from HIBP_BNN_multihead import HIBP_BNN

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None,
            single_head=True, task_inf=False, val=False, verbose=True, name='vcl', log_dir='logs',
            use_local_reparam=False, optimism=True, pred_ent=True, use_uncert=False,
            batch_size_entropy=None, ts_stop_gradients=False, ts=False, ts_cutoff=0.5, seed=100):
    assert not (single_head and task_inf), "Can't have both single head and task inference."
    x_testsets, y_testsets = [], []
    x_coresets, y_coresets = [], []

    all_acc, all_acc_ent = np.array([]), np.array([])
    all_uncerts = []

    for task_id in range(data_gen.max_iter):

        in_dim, out_dim = data_gen.get_dims()
        
        tf.compat.v1.reset_default_graph()

        if val:
            x_train, y_train, x_test, y_test, _, _ = data_gen.next_task()
        else:
            x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, 200, bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            stamp = None
            ml_model.close_session()

        # Train on non-coreset data
        mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances,
                           name="{0}_task{1}".format(name, task_id+1), tensorboard_dir=log_dir, use_local_reparam=use_local_reparam)

        # for coresets
        hparams = tf.contrib.training.HParams(name="{0}_task{1}".format(name, task_id+1), hidden_size=hidden_size, log_dir=log_dir, use_local_reparam=use_local_reparam)

        if os.path.isdir(mf_model.log_folder):
            print("Restoring model from {}".format(mf_model.log_folder))
            mf_model.restore(mf_model.log_folder)
        else:
            print("New model, training")
            mf_model.train(x_train, y_train, head, no_epochs, bsize, display_epoch=5, verbose=verbose)

        mf_weights, mf_variances = mf_model.get_weights()

        # get accuracies for all test sets seen so far
        acc = get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, bsize, single_head,
                         stamp, hparams, ibp=False, hibp=False)
        acc_ent, uncerts = get_scores_entropy(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, single_head,
                                     stamp, hparams, ibp=False, hibp=False, batch_size=batch_size_entropy,
                                     optimism=optimism, pred_ent=pred_ent, use_uncert=use_uncert)
        all_acc = concatenate_results(acc, all_acc)
        all_acc_ent = concatenate_results(acc_ent, all_acc_ent)
        all_uncerts.append(uncerts)
        mf_model.close_session()

    return [all_acc, all_acc_ent], all_uncerts

def run_vcl_ibp(hidden_size, alpha, no_epochs, data_gen,
                coreset_method, coreset_size, name,
                val, run_val_set=False, batch_size=None, single_head=False, task_inf=False,
                prior_mean=0.0, prior_var=1.0, alpha0=5.0,
                beta0 = 1.0, lambda_1 = 1.0, lambda_2 = 1.0, learning_rate=0.001,
                learning_rate_decay=0.87,
                no_pred_samples=100, ibp_samples = 10,
                log_dir='logs', tb_logging=True,
                use_local_reparam=False, implicit_beta=True,
                hibp=False,
                optimism=True, pred_ent=True, use_uncert=False, batch_size_entropy=None,
                ts_stop_gradients=False, ts=False, ts_cutoff=0.5, seed=100):

    assert not (single_head and task_inf), "Can't have both single head and task inference at the same time."
    all_acc, all_acc_ent = np.array([]), np.array([])
    all_uncerts, Zs = [], []
    x_testsets, y_testsets = [], []
    x_valsets, y_valsets = [], []
    x_coresets, y_coresets = [], []
    all_x_testsets, all_y_testsets = [], []

    for task_id in range(data_gen.max_iter):
        if val:
            _, _, x_test, y_test, _, _ = data_gen.next_task()
        else:
            _, _, x_test, y_test = data_gen.next_task()
        all_x_testsets.append(x_test)
        all_y_testsets.append(y_test)

    data_gen.reset_cur_iter()

    for task_id in range(data_gen.max_iter):

        in_dim, out_dim = data_gen.get_dims()

        tf.compat.v1.reset_default_graph()

        if val:
            x_train, y_train, x_test, y_test, x_val, y_val = data_gen.next_task()
            x_valsets.append(x_val)
            y_valsets.append(y_val)
        else:
            x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        if isinstance(no_epochs, list):
            n = no_epochs[task_id]
        else:
            n = no_epochs

        if isinstance(learning_rate, list):
            lr = learning_rate[task_id]
        else:
            lr = learning_rate

        if len(alpha) == 1:
            a = alpha[0]
        else:
            a = alpha[task_id]

        # Train network with maximum likelihood to initialize first model
        # lambda_1 --> temp of the variational Concrete posterior
        # lambda_2 --> temp of the relaxed prior, for task != 0 this should be lambda_1!!!
        if task_id == 0:
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, 200, bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            mf_betas = None
            stamp = {0: [0]*len(hidden_size)}
            ml_model.close_session()

        if hibp and len(hidden_size) > 1:
            model = HIBP_BNN(a, input_size=in_dim, hidden_size=hidden_size,
                             output_size=out_dim,
                             training_size=x_train.shape[0], num_ibp_samples=ibp_samples,
                             prev_means=mf_weights,
                             prev_log_variances=mf_variances, prev_betas=mf_betas,
                             stamp=stamp, # currently not used
                             alpha0=alpha0, beta0=beta0, learning_rate=lr,
                             learning_rate_decay=learning_rate_decay,
                             prior_mean=prior_mean, prior_var=prior_var, lambda_1=lambda_1,
                             lambda_2=lambda_2 if task_id == 0 else lambda_1,
                             no_pred_samples=no_pred_samples,
                             tensorboard_dir=log_dir,
                             tb_logging=tb_logging,
                             name='{0}_task{1}'.format(name, task_id + 1),
                             use_local_reparam=use_local_reparam,
                             implicit_beta=implicit_beta)
        else:
            model = IBP_BNN(in_dim, hidden_size, out_dim, x_train.shape[0], num_ibp_samples=ibp_samples,
                            prev_means=mf_weights,
                            prev_log_variances=mf_variances, prev_betas=mf_betas,
                            stamp=stamp,
                            alpha0=alpha0, beta0=beta0, learning_rate=lr,
                            learning_rate_decay=learning_rate_decay,
                            prior_mean=prior_mean, prior_var=prior_var, lambda_1=lambda_1,
                            lambda_2=lambda_2 if task_id == 0 else lambda_1,
                            no_pred_samples=no_pred_samples,
                            tensorboard_dir=log_dir,
                            tb_logging=tb_logging,
                            name='{0}_task{1}'.format(name, task_id + 1),
                            use_local_reparam=use_local_reparam,
                            implicit_beta=implicit_beta,
                            ts_stop_gradients=ts_stop_gradients, ts=ts,
                            seed=seed)

        # for coresets
        hparams = tf.contrib.training.HParams(a=a, hidden_size=hidden_size, ibp_samples=ibp_samples,
                 alpha0=alpha0, beta0=beta0, lr=lr, learning_rate_decay=learning_rate_decay,
                 prior_mean=prior_mean, prior_var=prior_var, lambda_1=lambda_1, lambda_2=lambda_2 if task_id == 0 else lambda_1,
                 no_pred_samples=no_pred_samples, log_dir=log_dir, tb_logging=tb_logging,
                 name='{0}_task{1}'.format(name, task_id + 1), use_local_reparam=use_local_reparam,
                 implicit_beta=implicit_beta)

        model.create_model()
        if os.path.isdir(model.log_folder):
            print("Restoring model: {}".format(model.log_folder))
            model.restore(model.log_folder)
        else:
            print("New model, training")
            model.train(x_train, y_train, head, n, bsize)

        mf_weights, mf_variances, mf_betas = model.get_weights() # stamp: dict task_id: list # list of len n_layers
        if ts:
            _, s_new = model.prediction_Zs(x_train, None, task_id, cut_off=ts_cutoff)
            stamp[task_id+1] = [max(s, s_old+2) for s, s_old in zip(s_new, stamp[task_id])]
        else:
            # just fill with something for placeholder - will not be used.
            stamp[task_id+1] = [s for s in stamp[task_id]]
        print("stamp: {}".format(stamp))

        if val and run_val_set:
            acc = get_scores(model, x_valsets, y_valsets, x_coresets, y_coresets, bsize, single_head, stamp,
                             hparams, ibp=not hibp, hibp=hibp)
            acc_ent, uncerts = get_scores_entropy(model, x_valsets, y_valsets, x_coresets, y_coresets, single_head, stamp,
                                         hparams, ibp=not hibp, hibp=hibp,
                                         batch_size=batch_size_entropy, optimism=optimism, pred_ent=pred_ent,
                                         use_uncert=use_uncert)
        else:
            acc = get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, bsize, single_head, stamp,
                             hparams, ibp=not hibp, hibp=hibp)
            acc_ent, uncerts = get_scores_entropy(model, x_testsets, y_testsets, x_coresets, y_coresets, single_head, stamp,
                                         hparams, ibp=not hibp, hibp=hibp,
                                         batch_size=batch_size_entropy, optimism=optimism, pred_ent=pred_ent,
                                         use_uncert=use_uncert)
        all_acc = concatenate_results(acc, all_acc)
        all_acc_ent = concatenate_results(acc_ent, all_acc_ent)
        Z, _ = model.prediction_Zs(x_test, bsize, task_id)
        Zs.append(Z)
        all_uncerts.append(uncerts)
        model.close_session()

    Zs = [item for sublist in Zs for item in sublist]
    return [all_acc, all_acc_ent], Zs, all_uncerts, stamp