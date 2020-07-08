import os
from statistics import variance
import pickle
import cv2
import numpy as np


def load_from_pickle(filepath):
    with open(filepath, 'rb') as fo:
        data_dict = pickle.load(fo)

    return data_dict


def save_to_pickle(filepath, data):
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)


def array2im(img, length):
    try:
        channel_len = int(length / 3)
        resolution = int(np.sqrt(channel_len))
        r = img[0:channel_len].reshape((resolution, resolution))
        g = img[channel_len: 2*channel_len].reshape((resolution, resolution))
        b = img[2*channel_len: 3*channel_len].reshape((resolution, resolution))
    except:
        return None
    # return cv2.merge((b, g, r))
    return cv2.merge((r, g, b))


def load_cifar10_training_data(dataset_dir):
    print('Loading training data...\n')
    f1 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_1'))
    f2 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_2'))
    f3 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_3'))
    f4 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_4'))
    f5 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_5'))

    train_x = np.concatenate((f1['data'], f2['data'], f3['data'], f4['data'], f5['data']), axis=0)
    train_y = np.concatenate((f1['labels'], f2['labels'], f3['labels'], f4['labels'], f5['labels']), axis=0)

    return train_x, train_y


def load_cifar10_test_data(dataset_dir):
    print('Loading test data...\n')
    f1 = load_from_pickle(os.path.join(dataset_dir, 'test_batch'))
    test_x = np.asarray(f1['data'])
    test_y = np.asarray(f1['labels'])

    return test_x, test_y


def extract_random_patches(num_patches, rf_size, data_x, image_dimensions):
    """ extract random patches from train set of CIFAR10 dataset """
    patches = np.zeros((num_patches, rf_size * rf_size * 3))
    for i in range(num_patches):
        if i % 10000 == 0:
            print("Extracting patch: {} / {}".format(i, num_patches))
        r = int(np.random.uniform(image_dimensions[0] - rf_size))
        c = int(np.random.uniform(image_dimensions[1] - rf_size))
        patch_ = array2im(data_x[i % data_x.shape[0], :], data_x.shape[1])
        patch = patch_[r:r + rf_size, c:c + rf_size, :]
        patches[i] = np.reshape(cv2.transpose(patch), -1, order='F')
    return np.transpose(patches)


def normalize_for_contrast(patches):
    """ normalize for contrast """
    patches_mean = np.reshape(np.mean(patches, 1), (1, -1))
    patches_variance = np.reshape(np.var(patches, 1, ddof=1), (1, -1))
    patches = np.transpose(patches) - patches_mean
    patches = patches / np.sqrt(patches_variance + 10)
    return patches

def data_whitening(patches):
    """ whiten """
    C = np.cov(patches, rowvar=False)  # 108 x 108 (for 6x6x3 kernels)
    M = np.mean(patches, axis=0)
    d, V = np.linalg.eig(C)
    D = np.diag(np.sqrt(1. / (d + 0.1)))
    P = np.matmul(np.matmul(V, D), V.T)
    patches = np.matmul(patches - M, P)

    return patches, M, P 

def data_whitening(patches):
    """ whiten """
    C = np.dot(patches.T, patches)
    d, V = np.linalg.eigh(C)
    D = np.diag(1. / np.sqrt(d + 0.1))
    W = np.dot(np.dot(V, D), V.T)
    patches = np.dot(patches, W)

    return patches, np.mean(np.mean(patches)), W


def im2col_sliding_strided(patch, window, stepsize=1):
    m, n = patch.shape
    s0, s1 = patch.strides
    nrows = m-window[0]+1
    ncols = n-window[1]+1
    shp = window[0], window[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(patch, shape=shp, strides=strd)
    return out_view.reshape(window[0]*window[1], -1)[:, ::stepsize]


