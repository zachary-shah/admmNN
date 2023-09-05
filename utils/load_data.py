"""
Datasets to load (MNIST, FMNIST, CIFAR-10)
"""

import numpy as np
from numpy.random import randn
from math import ceil
# File I/O
import pickle, gzip
import pandas as pd
from os.path import dirname, join, abspath

# The MNIST 
def load_mnist(dataset_rel_path=join('Convex-NN-Journal', 'Datasets', 'mnist.pkl.gz'), 
               n=3000, 
               downsample=False, 
               stride=3):
    
    project_root = dirname(abspath(''))
    load_path = join(project_root, dataset_rel_path)
    with gzip.open(load_path, 'rb') as fmnist:
        training_data, validation_data, test_data = pickle.load(fmnist, encoding="bytes")
    dim = ceil(28 / stride)

    X_training_raw = np.array(training_data[0])
    training_y = np.array(training_data[1])
    training_mask = (training_y == 2) | (training_y == 8)
    X_training_raw = X_training_raw[training_mask, :]
    training_y = np.sign(training_y[training_mask] - 5.)
    if downsample:
        training_X = np.zeros([training_y.size, dim ** 2])
        for i in range(training_y.size):
            x = X_training_raw[i, :].reshape([28, 28])
            x = x[::stride, ::stride];
            training_X[i, :] = x.reshape(dim ** 2)
    else:
        training_X = X_training_raw

    X_validation_raw = np.array(validation_data[0])
    validation_y = np.array(validation_data[1])
    validation_mask = (validation_y == 2) | (validation_y == 8)
    X_validation_raw = X_validation_raw[validation_mask, :]
    validation_y = np.sign(validation_y[validation_mask] - 5.)
    if downsample:
        validation_X = np.zeros([validation_y.size, dim ** 2])
        for i in range(validation_y.size):
            x = X_validation_raw[i, :].reshape([28, 28])
            x = x[::stride, 0::stride]
            validation_X[i, :] = x.reshape(dim ** 2)
    else:
        validation_X = X_validation_raw

    X_test_raw = np.array(test_data[0])
    test_y = np.array(test_data[1])
    test_mask = (test_y == 2) | (test_y == 8)
    X_test_raw = X_test_raw[test_mask, :]
    test_y = np.sign(test_y[test_mask] - 5.)
    if downsample:
        test_X = np.zeros([test_y.size, dim ** 2])
        for i in range(test_y.size):
            x = X_test_raw[i, :].reshape([28, 28])
            x = x[::stride, ::stride]
            test_X[i, :] = x.reshape(dim ** 2)
    else:
        test_X = X_test_raw

    training_X = np.concatenate((training_X, validation_X), axis=0)
    training_y = np.concatenate((training_y, validation_y), axis=0)
    training_X = training_X[:n, :]
    training_y = training_y[:n]

    return training_X, training_y.astype(int), test_X, test_y.astype(int)

# Fashion MNIST 
def load_fmnist(dataset_rel_path=join('Convex-NN-Journal', 'Datasets', 'Fashion MNIST'), 
                n=3000, 
                downsample=False, 
                stride=3, 
                normalize=True):
    
    project_root = dirname(abspath(''))
    path = join(project_root, dataset_rel_path)
    training_labels_path = join(path, 'train-labels-idx1-ubyte')
    training_images_path = join(path, 'train-images-idx3-ubyte')
    test_labels_path = join(path, 't10k-labels-idx1-ubyte')
    test_images_path = join(path, 't10k-images-idx3-ubyte')

    with open(training_labels_path, 'rb') as training_lbpath:
        training_y = np.frombuffer(training_lbpath.read(),
                                   dtype=np.uint8, offset=8)
    with open(training_images_path, 'rb') as training_imgpath:
        X_training_raw = np.frombuffer(training_imgpath.read(),
                                       dtype=np.uint8, offset=16).reshape(len(training_y), 784)
    with open(test_labels_path, 'rb') as test_lbpath:
        test_y = np.frombuffer(test_lbpath.read(),
                               dtype=np.uint8, offset=8)
    with open(test_images_path, 'rb') as test_imgpath:
        X_test_raw = np.frombuffer(test_imgpath.read(),
                                   dtype=np.uint8, offset=16).reshape(len(test_y), 784)
    dim = ceil(28 / stride)

    training_mask = (training_y == 2) | (training_y == 8)
    X_training_raw = X_training_raw[training_mask, :]
    training_y = np.sign(training_y[training_mask] - 5.)
    if downsample:
        training_X = np.zeros([training_y.size, dim ** 2])
        for i in range(training_y.size):
            x = X_training_raw[i, :].reshape([28, 28])
            x = x[::stride, ::stride]
            training_X[i, :] = x.reshape(dim ** 2)
    else:
        training_X = X_training_raw

    test_mask = (test_y == 2) | (test_y == 8)
    X_test_raw = X_test_raw[test_mask, :]
    test_y = np.sign(test_y[test_mask] - 5.)
    if downsample:
        test_X = np.zeros([test_y.size, dim ** 2])
        for i in range(test_y.size):
            x = X_test_raw[i, :].reshape([28, 28])
            x = x[::stride, ::stride]
            test_X[i, :] = x.reshape(dim ** 2)
    else:
        test_X = X_test_raw

    training_X = training_X[:n, :]
    training_y = training_y[:n]

    if normalize:
        X_mean, X_std = training_X.mean(), training_X.std()
        training_X = (training_X - X_mean) / (X_std + 1e-10)
        test_X = (test_X - X_mean) / (X_std + 1e-10)
    return training_X, training_y.astype(int), test_X, test_y.astype(int)

# The CIFAR-10 
def load_cifar(dataset_rel_path = join('datasets', 'cifar-10-batches-py'), 
               n=10000, 
               downsample=False, 
               binary_classes=True,
               stride=3, 
               normalize=True):
    
    project_root = dirname(abspath(''))
    path = join(project_root, dataset_rel_path)
    dim = ceil(32 / stride)

    dats_training = []
    for i in range(1, 6):
        training_file_name = 'data_batch_' + str(i)
        with open(join(path, training_file_name), 'rb') as ftrain:
            dats_training += [pickle.load(ftrain, encoding='latin1')]
    X_training_raw = np.concatenate([np.array(dat_training['data'])
                                     for dat_training in dats_training], axis=0)  # (50000, 3072)
    # X_training_raw = X_training_raw.reshape(10000, 3, 32, 32)
    training_y = np.concatenate([np.array(dat_training['labels'])
                                 for dat_training in dats_training], axis=0)  # (50000,)
    
    if binary_classes:
        training_mask = (training_y == 0) | (training_y == 1)
        X_training_raw = X_training_raw[training_mask, :]
        training_y = np.sign(training_y[training_mask] - 0.5)

    if downsample:
        training_X = np.zeros([training_y.size, 3 * (dim ** 2)])
        for i in range(training_y.size):
            x = X_training_raw[i, :].reshape([32, 32, 3])
            x = x[::stride, ::stride, :]
            training_X[i, :] = x.reshape(3 * (dim ** 2))
    else:
        training_X = X_training_raw

    test_file_name = 'test_batch'
    with open(join(path, test_file_name), 'rb') as ftest:
        dat_test = pickle.load(ftest, encoding='latin1')
    images = dat_test['data']
    labels = dat_test['labels']
    X_test_raw = np.array(images)  # (10000, 3072)
    test_y = np.array(labels)  # (10000,)

    if binary_classes:
        test_mask = (test_y == 0) | (test_y == 1)
        X_test_raw = X_test_raw[test_mask, :]
        test_y = np.sign(test_y[test_mask] - 0.5)
        
    if downsample:
        test_X = np.zeros([test_y.size, 3 * (dim ** 2)])
        for i in range(test_y.size):
            x = X_test_raw[i, :].reshape([32, 32, 3])
            x = x[::stride, ::stride, :]
            test_X[i, :] = x.reshape(3 * (dim ** 2))
    else:
        test_X = X_test_raw

    training_X = training_X[:n, :]
    training_y = training_y[:n]

    if normalize:
        X_mean, X_std = training_X.mean(), training_X.std()
        training_X = (training_X - X_mean) / (X_std + 1e-10)
        test_X = (test_X - X_mean) / (X_std + 1e-10)
    return training_X, training_y, test_X, test_y

# preprocess the already loaded CIFAR10
def preprocess_cifar10(X_train=[], 
                     X_test=[], 
                     Y_train=[], 
                     Y_test=[], 
                     n_train=10000,
                     n_test=1000, 
                     downsample=False, 
                     binary_classes=True,
                     stride=3, 
                     normalize=True):
    
    assert not (len(X_train) == 0 or len(X_test) == 0 or len(Y_train) == 0 or len(Y_test) == 0), "Must provide valid train/test CIFAR data. Retrieve using (X_train,Y_train),(X_test,Y_test) = keras.datasets.cifar10.load_data()"

    # limit classes only to class 0 and 1
    if binary_classes:
        training_mask = (Y_train == 0) | (Y_train == 1)
        training_mask = training_mask.flatten()
        X_train = X_train[training_mask, :, :, :]
        Y_train = np.sign(Y_train[training_mask] - 0.5)

        test_mask = (Y_test == 0) | (Y_test == 1)
        test_mask = test_mask.flatten()
        X_test = X_test[test_mask, :]
        Y_test = np.sign(Y_test[test_mask] - 0.5)

    # downsample images to 1/stride of their pixels
    if downsample:
        dim = ceil(32 / stride)
        X_train_proc = np.zeros([Y_train.size, 3 * (dim ** 2)])
        for i in range(Y_train.size):
            x = X_train[i, :][::stride, ::stride, :]
            X_train_proc[i, :] = x.reshape(3 * (dim ** 2))

        X_test_proc = np.zeros([Y_test.size, 3 * (dim ** 2)])
        for i in range(Y_test.size):
            x = X_test[i, :][::stride, ::stride, :]
            X_test_proc[i, :] = x.reshape(3 * (dim ** 2))
    else:
        X_train_proc = X_train.reshape(3 * 32**2)
        X_test_proc = X_test.reshape(3 * 32**2)

    X_train_proc = X_train_proc[:n_train, :]
    Y_train = Y_train[:n_train]

    X_test_proc = X_test_proc[:n_test, :]
    Y_test = Y_test[:n_test]

    if normalize:
        X_mean, X_std = X_train_proc.mean(), X_train_proc.std()
        X_train_proc = (X_train_proc - X_mean) / (X_std + 1e-10)
        X_test_proc = (X_test_proc - X_mean) / (X_std + 1e-10)

    return X_train_proc, Y_train.squeeze(), X_test_proc, Y_test.squeeze()

