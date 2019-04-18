import numpy as np
import math
from PIL import Image
from utils import array2im


def plot_centroids(**kwargs):
    centroids = kwargs.get("centroids", None)
    height = kwargs.get("height", None)
    width = kwargs.get("width", None)

    if height is None:
        height = width
    elif width is None:
        width = height

    if centroids is None or height is None or width is None:
        return None

    channels = centroids.shape[1] / (height * width)
    if channels != 3 and channels != 1:
        return None

    K = centroids.shape[0]

    cols = math.ceil(np.sqrt(K))
    rows = math.ceil(K / cols)
    count = cols * rows

    image = np.ones((int(rows*height), int(cols*width), int(channels)))

    for i in range(centroids.shape[0]):
        r = int(math.floor(i/cols))
        c = int(i % cols)
        current_centroid = array2im(centroids[i, :], centroids.shape[1])
        image[r*height: (r+1)*height, c*width: (c+1)*width, :] = current_centroid

    mn = image.min()
    mx = image.max()
    image = (image - mn) / (mx - mn)

    result = Image.fromarray((image * 255).astype(np.uint8))
    result.save('/ext/centroids_{}.png'.format(centroids.shape[0]))


