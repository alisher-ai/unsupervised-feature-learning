from extract_features import extract_features_from_data, extract_features_from_whitened_data, \
    extract_features_post_processing
import numpy as np
from sklearn import svm

from utils import load_cifar10_training_data, extract_random_patches, normalize_for_contrast, data_whitening, \
    load_cifar10_test_data
from run_kmeans import run_kmeans
from plot_centroids import plot_centroids

dataset_dir = "/ext/alisher_research/cifar-10-batches-py"

rf_size = 6
num_centroids = 400
whitening = False
num_patches = 40000
image_dimensions = [32, 32, 3]

train_x, train_y = load_cifar10_training_data(dataset_dir)
patches = extract_random_patches(num_patches, rf_size, train_x, train_y, image_dimensions)
patches = normalize_for_contrast(patches)
if whitening:
    patches, m, p = data_whitening(patches)


""" run k-means """
centroids = run_kmeans(patches, num_centroids, 50)
plot_centroids(centroids=centroids, width=rf_size, height=rf_size)

""" extract training features """
trainXCs = extract_features_post_processing(whitening, train_x, centroids, rf_size, image_dimensions)

""" Support Vector Machine (SVM) Classifier training """
clf = svm.SVC(kernel='rbf', verbose=True, max_iter=1000000, C=100)
clf.fit(trainXCs, train_y)
print(clf.score(trainXCs, train_y))

""" Testing the model on test data: """
test_x, test_y = load_cifar10_test_data(dataset_dir)

if whitening:
    testXC = extract_features_from_whitened_data(test_x, centroids, rf_size, image_dimensions, m, p)
else:
    testXC = extract_features_from_data(test_y, centroids, rf_size, image_dimensions)

testXCs = extract_features_post_processing(whitening, test_x, centroids, rf_size, image_dimensions)

testXCs = testXCs[:10, :]
y_pred = clf.predict(testXCs)
print(clf.score(testXCs, test_y))
