import numpy as np
from utils import array2im, im2col_sliding_strided, normalize_for_contrast


def extract_features_from_whitened_data(data_x, centroids, rf_size, image_dimensions, m, p):
    return None


def extract_features_from_data(data_x, centroids, rf_size, image_dimensions):
    num_centroids = centroids.shape[0]

    data_x_centroids = np.zeros((data_x.shape[0], num_centroids * 4))

    for i, x in enumerate(data_x):
        if i % 1000 == 0:
                print("Extracting features: {} / {}".format(i, data_x.shape[0]))
        patches = np.transpose(np.concatenate((im2col_sliding_strided(data_x[i, :1024].reshape(32, 32), window=[6, 6]),
                                               im2col_sliding_strided(data_x[i, 1024:2048].reshape(32, 32), window=[6, 6]),
                                               im2col_sliding_strided(data_x[i, 2048:].reshape(32, 32), window=[6, 6])), axis=0))

        """ normalize for contrast
        """
        patches = np.transpose(normalize_for_contrast(patches))

        """ compute 'triangle' activation function
        """
        xx = np.sum(np.power(patches, 2), axis=1)
        cc = np.sum(np.power(centroids, 2), axis=1)
        xc = np.matmul(patches, np.transpose(centroids))

        z1 = np.reshape(xx, (-1, 1)) - 2 * xc
        z2 = z1 + np.reshape(cc, (1, -1))
        z = np.sqrt(z2)

        val, inds = [z.min(1), z.argmin(1)]

        """ average distance to centroids for each patch
        """
        mu = np.mean(z, 1)

        """ patches is now the data matrix of activations for each patch
        """
        patches = np.reshape(mu, (-1, 1)) - z
        patches[np.where(patches < 0)] = 0

        prows = image_dimensions[0] - rf_size + 1
        pcols = image_dimensions[1] - rf_size + 1

        patches = patches.reshape((prows, pcols, num_centroids))
        halfr = int(round(float(prows) / 2))
        halfc = int(round(float(pcols) / 2))

        q1 = sum(sum(patches[:halfr, :halfc, :]))
        q2 = sum(sum(patches[halfr:, :halfc, :]))
        q3 = sum(sum(patches[:halfr, halfc:, :]))
        q4 = sum(sum(patches[halfr:, halfc:, :]))

        data_x_centroids[i, :] = np.concatenate((q1, q2, q3, q4), axis=0)

    return data_x_centroids


def extract_features_post_processing(whitening, data_x, centroids, rf_size, image_dimensions, m=None, p=None):
    if whitening:
        trainXC = extract_features_from_whitened_data(data_x, centroids, rf_size, image_dimensions, m, p)
    else:
        trainXC = extract_features_from_data(data_x, centroids, rf_size, image_dimensions)

    trainXC_mean = np.mean(trainXC, 0)
    trainXC_sd = np.sqrt(np.var(trainXC, 0, ddof=1) + 0.01)

    trainXCs = np.transpose((trainXC - trainXC_mean) / trainXC_sd)
    trainXCs = np.concatenate((np.transpose(trainXCs), np.ones((trainXCs.shape[1], 1))), axis=1)

    # trainXCs = trainXCs[:10, :]

    return trainXCs