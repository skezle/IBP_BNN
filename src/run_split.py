import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
import pdb
import os.path
import argparse
sys.path.extend(['alg/'])
from vcl import run_vcl, run_vcl_ibp
import coreset
from copy import deepcopy

from skimage.color import rgb2gray


class SplitMnistGenerator:
    def __init__(self, val=False, cl3=False):
        # train, val, test (50000, 784) (10000, 784) (10000, 784)
        self.val = val
        self.cl3 = cl3
        # data already shuffled
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        if self.val:
            self.X_train = train_set[0]
            self.X_val = valid_set[0]
            self.train_label = train_set[1]
            self.val_label = valid_set[1]
        else:
            self.X_train = np.vstack((train_set[0], valid_set[0]))
            self.train_label = np.hstack((train_set[1], valid_set[1]))

        self.X_test = test_set[0]
        self.test_label = test_set[1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]

        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        if self.cl3:
            return self.X_train.shape[1], 10
        else:
            return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            if self.cl3:
                n, d = self.get_dims()
                next_y_train_1d = np.copy(next_y_train)
                next_y_train = np.zeros((next_y_train_1d.shape[0], d))
                next_y_train[:, self.cur_iter * 2] = next_y_train_1d.reshape(-1)
                next_y_train[:, self.cur_iter * 2 + 1] = 1 - next_y_train_1d.reshape(-1)
                assert next_y_train.shape[1] == d
            else:
                next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1)))) # N x 1
            if self.cl3:
                next_y_test_1d = np.copy(next_y_test)
                next_y_test = np.zeros((next_y_test_1d.shape[0], d))
                next_y_test[:, self.cur_iter * 2] = next_y_test_1d.reshape(-1)
                next_y_test[:, self.cur_iter * 2 + 1] = 1 - next_y_test_1d.reshape(-1)
            else:
                next_y_test = np.hstack((next_y_test, 1 - next_y_test))

            if self.val:
                val_0_id = np.where(self.val_label == self.sets_0[self.cur_iter])[0]
                val_1_id = np.where(self.val_label == self.sets_1[self.cur_iter])[0]
                next_x_val = np.vstack((self.X_val[val_0_id], self.X_val[val_1_id]))

                next_y_val = np.vstack((np.ones((val_0_id.shape[0], 1)), np.zeros((val_1_id.shape[0], 1))))
                if self.cl3:
                    next_y_val_1d = np.copy(next_y_val)
                    next_y_val = np.zeros((next_y_val_1d.shape[0], d))
                    next_y_val[:, self.cur_iter * 2] = next_y_val_1d.reshape(-1)
                    next_y_val[:, self.cur_iter * 2 + 1] = 1 - next_y_val_1d.reshape(-1)
                else:
                    next_y_val = np.hstack((next_y_val, 1 - next_y_val))
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test, next_x_val, next_y_val
            else:
                self.cur_iter += 1
                return next_x_train, next_y_train, next_x_test, next_y_test

    def reset_cur_iter(self):
        self.cur_iter = 0

    def set_cur_iter(self, ind):
        self.cur_iter = ind


class SplitMnistImagesGenerator(SplitMnistGenerator):
    """ Thanks https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits
    """
    def __init__(self, val=False, cl3=False):

        super(SplitMnistImagesGenerator, self).__init__(val=val, cl3=cl3)

        # 12000 test, 52000 train
        # datasets already shuffled
        train = np.loadtxt('data/mnist_background_images/mnist_background_images_train.amat')
        test = np.loadtxt('data/mnist_background_images/mnist_background_images_test.amat')
        all = np.vstack((train, test))
        if self.val:
            n_train = 40000
            n_test = 10000
            n_val = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:(n_train+n_test), :-1]
            self.test_label = all[n_train:(n_train+n_test), -1:]
            self.X_val = all[(n_train+n_test):, :-1]
            self.val_label = all[(n_train+n_test):, -1:]
        else:
            n_train = 52000
            n_test = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:, :-1]
            self.test_label = all[n_train:, -1:]

        assert self.X_train.shape[1] == 28 * 28
        assert self.X_test.shape[1] == 28 * 28

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0


class SplitMnistRandomGenerator(SplitMnistGenerator):
    """ Thanks https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits

    """
    def __init__(self, val=False, cl3=False):

        super(SplitMnistRandomGenerator, self).__init__(val=val, cl3=cl3)

        # 12000 test, 50000 train
        # datasets already shuffled
        train = np.loadtxt('data/mnist_background_random/mnist_background_random_train.amat')
        test = np.loadtxt('data/mnist_background_random/mnist_background_random_test.amat')
        all = np.vstack((train, test))
        if self.val:
            n_train = 40000
            n_test = 10000
            n_val = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:(n_train+n_test), :-1]
            self.test_label = all[n_train:(n_train+n_test), -1:]
            self.X_val = all[(n_train+n_test):, :-1]
            self.val_label = all[(n_train+n_test):, -1:]
        else:
            n_train = 52000
            n_test = 10000
            self.X_train = all[:n_train, :-1]
            self.train_label = all[:n_train, -1:]
            self.X_test = all[n_train:, :-1]
            self.test_label = all[n_train:, -1:]

        assert self.X_train.shape[1] == 28 * 28
        assert self.X_test.shape[1] == 28 * 28

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

class SplitCIFAR10Generator(SplitMnistGenerator):
    def __init__(self, val=False, cl3=False):

        super(SplitCIFAR10Generator, self).__init__(val=val, cl3=cl3)
        # train, val, test (40000, 3072) (10000, 3072) (10000, 3072)
        self.val = val
        self.cl3 = cl3

        train, test = tf.compat.v1.keras.datasets.cifar10.load_data() # (50000, 32, 32, 3), (50000, 1), (10000, 32, 32, 3), (10000, 1)
        train_labels = train[1].reshape(-1)
        train = train[0].reshape(-1, 32*32*3)
        test_labels = test[1].reshape(-1)
        test = test[0].reshape(-1, 32*32*3)

        train = train.astype('float32') / 255.0
        test = test.astype('float32') / 255.0

        if self.val:
            idx = np.random.permutation(train.shape[0])
            self.X_train = train[idx[10000:], :]
            self.X_val = train[idx[:10000], :]
            self.train_label = train_labels[idx[10000:]]
            self.val_label = train_labels[idx[:10000]]
        else:
            self.X_train = train
            self.train_label = train_labels

        self.X_test = test
        self.test_label = test_labels

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]

        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

class SplitMix(SplitMnistGenerator):
    def __init__(self, val=False, cl3=False):
        super(SplitMix, self).__init__(val=val, cl3=cl3)

        # TODO: cl3

        # Get MNIST data and pad to make 32x32
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')  # train, val, test (50000, 784) (10000, 784) (10000, 784)

        if self.val:
            X_train_mnist = train_set[0].reshape((-1, 28, 28))
            X_val_mnist = valid_set[0].reshape((-1, 28, 28))
            train_label_mnist = train_set[1]
            val_label_mnist = valid_set[1]
        else:
            X_train_mnist = np.vstack((train_set[0], valid_set[0])).reshape((-1, 28, 28))
            train_label_mnist = np.hstack((train_set[1], valid_set[1]))

        X_test_mnist = test_set[0].reshape((-1, 28, 28))
        test_label_mnist = test_set[1]

        # zero padding
        _X_train_mnist = np.zeros((X_train_mnist.shape[0], 32, 32))
        _X_test_mnist = np.zeros((X_test_mnist.shape[0], 32, 32))
        _X_train_mnist[:, 2:30, 2:30] = X_train_mnist
        _X_test_mnist[:, 2:30, 2:30] = X_test_mnist
        _X_train_mnist = _X_train_mnist.reshape(-1, 32*32)
        _X_test_mnist = _X_test_mnist.reshape(-1, 32*32)

        if self.val:
            _X_val_mnist = np.zeros((X_val_mnist.shape[0], 32, 32))
            _X_val_mnist[:, 2:30, 2:30] = X_val_mnist
            _X_val_mnist = _X_val_mnist.reshape(-1, 32 * 32)

        # Get FMNIST data
        # Train: 60,000
        # Test: 10,000
        offset = 10
        train, test = tf.compat.v1.keras.datasets.fashion_mnist.load_data()
        train_labels = train[1].reshape(-1) + offset
        train = train[0]
        test_labels = test[1].reshape(-1) + offset
        test = test[0]

        train = train.astype('float32') / 255.0
        test = test.astype('float32') / 255.0

        X_test_fmnist = test
        test_label_fmnist = test_labels

        if self.val:
            idx = np.random.permutation(train.shape[0])
            X_val_fmnist = train[idx[:10000], :, :]
            val_label_fmnist = train_labels[idx[:10000]]
            X_train_fmnist = train[idx[10000:], :, :]
            train_label_fmnist = train_labels[idx[10000:]]
        else:
            X_train_fmnist = train
            train_label_fmnist = train_labels

        # zero padding
        _X_train_fmnist = np.zeros((X_train_fmnist.shape[0], 32, 32))
        _X_test_fmnist = np.zeros((X_test_fmnist.shape[0], 32, 32))
        _X_train_fmnist[:, 2:30, 2:30] = train
        _X_test_fmnist[:, 2:30, 2:30] = test
        _X_train_fmnist = _X_train_fmnist.reshape(-1, 32 * 32)
        _X_test_fmnist = _X_test_fmnist.reshape(-1, 32 * 32)

        if self.val:
            _X_val_fmnist = np.zeros((X_val_fmnist.shape[0], 32, 32))
            _X_val_fmnist[:, 2:30, 2:30] = X_val_fmnist
            _X_val_fmnist = _X_val_fmnist.reshape(-1, 32 * 32)

        # Get CIFAR data and transform to grey scale
        offset = 20
        train, test = tf.compat.v1.keras.datasets.cifar10.load_data()  # (50000, 32, 32, 3), (50000, 1), (10000, 32, 32, 3), (10000, 1)
        train_labels = train[1].reshape(-1) + offset
        train = rgb2gray(train[0]).reshape(-1, 32 * 32)
        test_labels = test[1].reshape(-1) + offset
        test = rgb2gray(test[0]).reshape(-1, 32 * 32)

        train = train.astype('float32') / 255.0
        test = test.astype('float32') / 255.0

        if self.val:
            idx = np.random.permutation(train.shape[0])
            X_train_cifar = train[idx[10000:], :]
            X_val_cifar = train[idx[:10000], :]
            train_label_cifar = train_labels[idx[10000:]]
            val_label_cifar = train_labels[idx[:10000]]
        else:
            X_train_cifar = train
            train_label_cifar = train_labels

        X_test_cifar = test
        test_label_cifar = test_labels

        # merge
        self.X_train = np.vstack((_X_train_mnist, _X_train_fmnist, X_train_cifar))
        self.train_label = np.hstack((train_label_mnist, train_label_fmnist, train_label_cifar))
        self.X_test = np.vstack((_X_test_mnist, _X_test_fmnist, X_test_cifar))
        self.test_label = np.hstack((test_label_mnist, test_label_fmnist, test_label_cifar))

        if self.val:
            self.X_val = np.vstack((_X_val_mnist, _X_val_fmnist, X_val_cifar))
            self.val_label = np.hstack((val_label_mnist, val_label_fmnist, val_label_cifar))

        self.sets_0 = [6, 8, 10, 16, 20, 22]
        self.sets_1 = [7, 9, 11, 17, 21, 23]

        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        if self.cl3:
            return self.X_train.shape[1], 12
        else:
            return self.X_train.shape[1], 2



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--single_head', action='store_true',
                        default=False,
                        dest='single_head',
                        help='Whether to use a single head.')
    parser.add_argument('--cl2', action='store_true',
                        default=False,
                        dest='cl2',
                        help='Whether to use a perform CL2: domain incremental learning.')
    parser.add_argument('--cl3', action='store_true',
                        default=False,
                        dest='cl3',
                        help='Whether to use a perform CL3: class incremental learning.')
    parser.add_argument('--num_layers', action='store',
                        dest='num_layers',
                        default=1,
                        type=int,
                        help='Number of layers in the NNs.')
    parser.add_argument('--runs', action='store',
                        dest='runs',
                        default=1,
                        type=int,
                        help='Number runs to perform.')
    parser.add_argument('--log_dir', action='store',
                        dest='log_dir',
                        default='logs',
                        help='TB log directory.')
    parser.add_argument('--dataset', action='store',
                        dest='dataset',
                        help='Which dataset to choose {normal, random, images, cifar10, mix}.')
    parser.add_argument('--tag', action='store',
                        dest='tag',
                        help='Tag to use in naming file outputs')
    parser.add_argument('--new_tag', action='store',
                        dest='new_tag',
                        default='',
                        help='New tag to use to store pickle file if we are reloading a chackpoint with the tag arg.')
    parser.add_argument('--use_local_reparam', action='store_true',
                        default=False,
                        dest='use_local_reparam',
                        help='Whether to use local reparam.')
    parser.add_argument('--alpha0', action='store',
                        default=5.0,
                        type=float,
                        dest='alpha0',
                        help='The prior and initialisation for the beta concentration param.')
    parser.add_argument('--hibp', action='store_true',
                        default=False,
                        dest='hibp',
                        help='Whether to use hibp.')
    parser.add_argument('--run_baselines', action='store_true',
                        default=False,
                        dest='run_baselines',
                        help='Whether to run the baselines.')
    parser.add_argument('--no_ibp', action='store_true',
                        default=False,
                        dest='no_ibp',
                        help='Whether not to run ibp.')
    parser.add_argument('--h', nargs='+',
                        dest='h_list',
                        type=int,
                        default=[5, 50],
                        help='List of hidden states')
    parser.add_argument('--beta_1', nargs='+',
                        dest='beta_1',
                        type=int,
                        default=1,
                        help='List KL gauss coefficients.')
    parser.add_argument('--K', action='store',
                        dest='K',
                        default=100,
                        type=int,
                        help='Variational truncation param for IBP.')
    parser.add_argument('--alpha',  nargs='+',
                        dest='alpha',
                        type=int,
                        default=[4],
                        help='H-IBP hyperparam.')
    parser.add_argument('--optimism', action='store_true',
                        default=False,
                        dest='optimism',
                        help='Whether to use optimism in the face of uncertainty when infering task head for CL2 and CL3.')
    parser.add_argument('--mutual_info', action='store_true',
                        default=False,
                        dest='mutual_info',
                        help='Whether to use predictive entropy or mutual information as a measure of uncertainty for task inference in CL2 and CL3.')
    parser.add_argument('--use_uncert', action='store_true',
                        default=False,
                        dest='use_uncert',
                        help='Whether the uncertainties of the uncertainties to help make choices for inferring CL2 and CL3.')
    parser.add_argument('--rand_coreset', action='store_true',
                        default=False,
                        dest='rand_coreset',
                        help='Whether to use a random coreset.')
    parser.add_argument('--batch_entropy', action='store_true',
                        default=False,
                        dest='batch_entropy',
                        help='Whether to use batches when calculating uncertainties for cl2 and cl3.')

    args = parser.parse_args()

    print('cl2                  = {!r}'.format(args.cl2))
    print('cl3                  = {!r}'.format(args.cl3))
    print('num_layers           = {!r}'.format(args.num_layers))
    print('runs                 = {!r}'.format(args.runs))
    print('alpha0               = {!r}'.format(args.alpha0))
    print('log_dir              = {!r}'.format(args.log_dir))
    print('dataset              = {!r}'.format(args.dataset))
    print('use_local_reparam    = {!r}'.format(args.use_local_reparam))
    print('hibp                 = {!r}'.format(args.hibp))
    print('run_baselines        = {!r}'.format(args.run_baselines))
    print('h_list               = {!r}'.format(args.h_list))
    print('K                    = {!r}'.format(args.K))
    print('tag                  = {!r}'.format(args.tag))
    print('new_tag              = {!r}'.format(args.new_tag))
    print('alpha                = {!r}'.format(args.alpha))
    print('no_ibp               = {!r}'.format(args.no_ibp))
    print('beta_1               = {!r}'.format(args.beta_1))
    print('optimism             = {!r}'.format(args.optimism))
    print('mutual_info          = {!r}'.format(args.mutual_info))
    print('use_uncert           = {!r}'.format(args.use_uncert))
    print('rand_coreset         = {!r}'.format(args.rand_coreset))

    seeds = list(range(1, 1 + args.runs))

    # We don't need a validation set
    val = False
    single_head = False
    task_inf = True if args.cl3 or args.cl2 else False
    assert not (args.cl3 and args.cl2), "Can't have both cl2 and cl3."

    def get_datagen():
        if args.dataset == 'normal':
            data_gen = SplitMnistGenerator(val=val, cl3=args.cl3)
        elif args.dataset == 'random':
            data_gen = SplitMnistRandomGenerator(val=val, cl3=args.cl3)
        elif args.dataset == 'images':
            data_gen = SplitMnistImagesGenerator(val=val, cl3=args.cl3)
        elif args.dataset == 'cifar10':
            data_gen = SplitCIFAR10Generator(val=val, cl3=args.cl3)
        elif args.dataset == 'mix':
            data_gen = SplitMix(val=val, cl3=args.cl3)
        else:
            raise ValueError('Pick dataset in {normal, random, images, cifar10}')
        return data_gen

    num_tasks = get_datagen().max_iter

    vcl_ibp_accs = np.zeros((2, len(seeds), num_tasks, num_tasks))  # 2 for cl1 and cl2 results
    baseline_accs = {h: np.zeros((2, len(seeds), num_tasks, num_tasks)) for h in args.h_list}
    all_ibp_uncerts = np.zeros((len(seeds), num_tasks, num_tasks))
    baseline_uncerts = {h: np.zeros((len(seeds), num_tasks, num_tasks)) for h in args.h_list}
    all_Zs = []

    # IBP params
    alpha0 = args.alpha0
    beta0 = 1.0
    lambda_1 = 0.7 # posterior
    lambda_2 = 0.7 # prior
    alpha = args.alpha
    # Gaussian prior params
    prior_mean = 0.0
    prior_var = 0.7
    # Other params
    batch_size = 512
    batch_size_entropy = 1500 if args.batch_entropy else None
    no_epochs = 600
    ibp_samples = 10
    no_pred_samples = 100
    # Coreset params
    coreset_size = 500 if args.rand_coreset else 0
    coreset_method = coreset.rand_from_batch if args.rand_coreset else lambda a: a

    for i in range(len(seeds)):
        s = seeds[i]
        tf.set_random_seed(s)
        np.random.seed(1)

        if not args.no_ibp:
            data_gen = get_datagen()
            name = "split_{0}_run{1}_{2}".format(args.dataset, i + 1, args.tag)
            hidden_size = [args.K] * args.num_layers
            # Z matrix for each task is output
            # This is overwritten for each run
            ibp_acc, Zs, _ = run_vcl_ibp(hidden_size=hidden_size, alpha=alpha,
                                         no_epochs= [int(no_epochs*1.2)] + [no_epochs]*(num_tasks-1),
                                         data_gen=data_gen, coreset_method=coreset_method,
                                         coreset_size=coreset_size,
                                         name=name, val=val, batch_size=batch_size,
                                         single_head=args.single_head, task_inf=task_inf,
                                         prior_mean=prior_mean, prior_var=prior_var, alpha0=alpha0,
                                         beta0=beta0, lambda_1=lambda_1, lambda_2=lambda_2,
                                         learning_rate=[0.001]*num_tasks,
                                         no_pred_samples=no_pred_samples, ibp_samples=ibp_samples, log_dir=args.log_dir,
                                         use_local_reparam=args.use_local_reparam,
                                         implicit_beta=True, hibp=args.hibp, beta_1=args.beta_1,
                                         optimism=args.optimism, pred_ent=False if args.mutual_info else True,
                                         use_uncert=args.use_uncert, batch_size_entropy=batch_size_entropy,
                                         seed=s)

            all_Zs.append(Zs)
            vcl_ibp_accs[0, i, :, :] = ibp_acc[0] # task known
            vcl_ibp_accs[1, i, :, :] = ibp_acc[1] # task infered

        if args.run_baselines:
            # Run Vanilla VCL
            # Comparison with other single layer neural networks
            for h in args.h_list:
                tf.reset_default_graph()
                hidden_size = [h] * args.num_layers
                data_gen = get_datagen()
                vcl_result, _ = run_vcl(hidden_size, no_epochs, data_gen,
                                        coreset_method, coreset_size, batch_size,
                                        single_head, task_inf, val=val,
                                        name='vcl_h{0}_{1}_run{2}_{3}'.format(h, args.dataset, i+1, args.tag),
                                        log_dir=args.log_dir, use_local_reparam=args.use_local_reparam,
                                        optimism=args.optimism, pred_ent=False if args.mutual_info else True,
                                        use_uncert=args.use_uncert, batch_size_entropy=batch_size_entropy)

                baseline_accs[h][0, i, :, :] = vcl_result[0]
                baseline_accs[h][1, i, :, :] = vcl_result[1]

    with open('results/split_mnist_{0}_{1}.pkl'.format(args.tag, args.new_tag), 'wb') as input_file:
        pickle.dump({'vcl_ibp': vcl_ibp_accs,
                     'vcl_baselines': baseline_accs,
                     'uncerts_ibp': all_ibp_uncerts,
                     'uncerts_vcl_baselines': baseline_uncerts,
                     'Z': all_Zs}, input_file)

    print("Finished running.")
