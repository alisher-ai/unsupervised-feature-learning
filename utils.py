import cPickle
import os
import numpy as np


def load_from_pickle(filepath):
    with open(filepath, 'rb') as fo:
        data_dict = cPickle.load(fo)

    return data_dict


def save_to_pickle(filepath, data):
    with open(filepath, 'wb') as fp:
        cPickle.dump(data, fp)


def load_cifar10_training_data(dataset_dir):
    print('Loading training data...\n')
    f1 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_1'))
    f2 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_2'))
    f3 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_3'))
    f4 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_4'))
    f5 = load_from_pickle(os.path.join(dataset_dir, 'data_batch_5'))

    train_x = np.concatenate((f1['data'], f2['data'], f3['data'], f4['data'], f5['data']), axis=0)
    train_y = np.concatenate((f1['labels'], f2['labels'], f3['labels'], f4['labels'], f5['labels']), axis=0)

    # del f1, f2, f3, f4, f5
    return train_x, train_y

def extract_random_patches(num_patches, rf_size, train_x, train_y, image_dimensions):
    patches = np.zeros((num_patches, rf_size * rf_size * 3))

    """ extract random patches from train set of CIFAR10 dataset
    """
    for i in range(num_patches):
        if i % 10000 == 0:
            print("Extracting patch: {} / {}".format(i, num_patches))
        r = int(np.random.uniform(image_dimensions[0] - rf_size + 1))
        c = int(np.random.uniform(image_dimensions[1] - rf_size + 1))
        patch = np.reshape(train_x[i % train_x.shape[0], :], tuple(image_dimensions))
        patch = patch[r:r + rf_size, c:c + rf_size, :]
        patches[i] = np.reshape(patch, -1)
    return patches


def normalize_for_contrast(patches):
    """ normalize for contrast
    """
    patches_mean = np.reshape(np.mean(patches, 1), (-1, 1))
    patches_variance = np.reshape(np.var(patches, 1), (-1, 1))
    patches = patches - patches_mean
    patches = patches / np.sqrt(patches_variance + 10)
    return patches


def data_whitening(patches):
    """ whiten
    """
    C = np.dot(patches.T, patches)
    d, V = np.linalg.eigh(C)
    D = np.diag(1. / np.sqrt(d + 0.1))
    W = np.dot(np.dot(V, D), V.T)
    patches = np.dot(patches, W)

    return patches