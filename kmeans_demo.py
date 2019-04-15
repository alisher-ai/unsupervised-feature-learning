import os

import numpy as np

from utils import load_cifar10_training_data, extract_random_patches, normalize_for_contrast, data_whitening
from run_kmeans import run_kmeans
from plot_centroids import plot_centroids

dataset_dir = "/ext/alisher_research/cifar-10-batches-py"

rf_size = 6
num_centroids = 1600
whitening = True
num_patches = 40000
image_dimensions = [32, 32, 3]

train_x, train_y = load_cifar10_training_data(dataset_dir)
patches = extract_random_patches(num_patches, rf_size, train_x, train_y, image_dimensions)
patches = normalize_for_contrast(patches)
if whitening:
    patches = data_whitening(patches)


""" run k-means
"""
centroids = run_kmeans(patches, num_centroids, 50)
# plot_centroids(centroids, rf_size, savefile)


"""


% run K-means
centroids = run_kmeans(patches, numCentroids, 50);
show_centroids(centroids, rfSize); drawnow;

% extract training features
if (whitening)
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM, M,P);
else
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM);
end

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

% train classifier using SVM
C = 100;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%

%% Load CIFAR test data
fprintf('Loading test data...\n');
f1=load([CIFAR_DIR '/test_batch.mat']);
testX = double(f1.data);
testY = double(f1.labels) + 1;
clear f1;

% compute testing features and standardize
if (whitening)
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
else
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
end
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));



"""
