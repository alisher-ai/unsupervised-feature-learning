import numpy as np
from scipy import sparse
import scipy.io as sio


def run_kmeans(patches, num_centroids, iterations):
    patches2 = np.sum(np.power(patches, 2), axis=1)
    centroids = np.random.randn(num_centroids, patches.shape[1]) * 0.1
    batch_size = 1000

    for itr in range(iterations):
        print("K-means iteration: {} / {}".format(itr, iterations))
        c2 = 0.5 * np.sum(np.power(centroids, 2), axis=1)

        summation = np.zeros((num_centroids, patches.shape[1]))
        counts = np.zeros((num_centroids, 1))
        loss = 0

        for i in range(0, patches.shape[0], batch_size):
            last_index = min(i + batch_size, patches.shape[0])
            m = last_index - i

            matrix = np.matmul(centroids, np.transpose(patches[i:last_index, :])) - np.reshape(c2, [-1, 1])
            [val, labels] = [matrix.max(0), matrix.argmax(0)]
            loss += np.sum(0.5 * patches2[i: last_index] - np.transpose(val))

            S = sparse.csr_matrix((np.ones(m), (np.array(range(m)), labels)), [m, num_centroids])
            summation = summation + np.transpose(S) * patches[i:last_index, :]
            counts += np.sum(S, axis=0).transpose()

        centroids = summation / counts
        bad_index = np.where(counts == 0)[0]
        centroids[bad_index, :] = 0

    return centroids
