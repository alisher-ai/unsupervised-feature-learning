from joblib import dump, load
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import svm
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.models import load_model

from extract_features import extract_features_from_data, extract_features_from_whitened_data, \
    extract_features_post_processing
from plot_centroids import plot_centroids
from run_kmeans import run_kmeans
from utils import load_cifar10_training_data, extract_random_patches, normalize_for_contrast, data_whitening, \
    load_cifar10_test_data

""" configs """
dataset_dir = "/ext/alisher_research/cifar-10-batches-py"
rf_size = 6
num_centroids = 1600
whitening = False
num_patches = 400000
image_dimensions = [32, 32, 3]
training_algorithm = 'keras_nn'  # sklearn_svm, keras_nn

""" load and extract the tiny patches from training data """
train_x, train_y = load_cifar10_training_data(dataset_dir)
# patches = extract_random_patches(num_patches, rf_size, train_x, image_dimensions)
# patches = normalize_for_contrast(patches)
# if whitening:
#     patches, m, p = data_whitening(patches)
#
# """ run k-means """
# centroids = run_kmeans(patches, num_centroids, 50)
# plot_centroids(centroids=centroids, width=rf_size, height=rf_size)
#
# """ extract training features """
# if whitening:
#     trainXC = extract_features_from_whitened_data(train_x, centroids, rf_size, image_dimensions, m, p)
# else:
#     trainXC = extract_features_from_data(train_x, centroids, rf_size, image_dimensions)
# trainXCs = extract_features_post_processing(trainXC, training_algorithm)
#
# dump(centroids, '/ext/alisher_research/centroids_{}.joblib'.format(num_centroids))
# dump(trainXCs, '/ext/alisher_research/trainXCs.joblib'.format(trainXCs))

centroids = load('/ext/alisher_research/centroids_{}.joblib'.format(num_centroids))
trainXCs = load('/ext/alisher_research/trainXCs.joblib')

""" Training process:
1. Support Vector Machine (SVM) Classifier training 
2. Keras single layer NN with ReLU activation function
"""
if training_algorithm == 'sklearn_svm':
    clf = svm.SVC(kernel='rbf', verbose=True, C=100, tol=0.001)
    clf.fit(trainXCs, train_y)
    dump(clf, '/ext/svm_model_{}_{}.joblib'.format(num_centroids, num_patches))
    print(clf.score(trainXCs, train_y))
elif training_algorithm == 'keras_nn':
    model = Sequential()
    # model.add(Dense(10, input_dim=trainXCs.shape[1], activation='tanh', W_regularizer=regularizers.l2(0.01)))
    model.add(Dense(100, input_dim=trainXCs.shape[1], activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    train_y_one_hot = np_utils.to_categorical(train_y)
    mcp_save = ModelCheckpoint('/ext/alisher_research/2layer_{}centroids_{}patches.h5'.format(num_centroids, num_patches), save_best_only=True, monitor='val_loss', mode='min')
    model.fit(trainXCs, train_y_one_hot, batch_size=1000, epochs=100, callbacks=[mcp_save])
    model.save('/ext/alisher_research/2layer_{}centroids_{}patches_final.h5'.format(num_centroids, num_patches))

""" Testing the model on test data: """
test_x, test_y = load_cifar10_test_data(dataset_dir)

if whitening:
    testXC = extract_features_from_whitened_data(test_x, centroids, rf_size, image_dimensions, m, p)
else:
    testXC = extract_features_from_data(test_x, centroids, rf_size, image_dimensions)
testXCs = extract_features_post_processing(testXC, training_algorithm)

if training_algorithm == 'sklearn_svm':
    clf = load('/ext/svm_model_{}_{}.joblib'.format(num_centroids, num_patches))
    y_pred = clf.predict(testXCs)
    print(clf.score(testXCs, test_y))
elif training_algorithm == 'keras_nn':
    model = load_model('/ext/alisher_research/2layer_{}centroids_{}patches_final.h5'.format(num_centroids, num_patches))

    test_y_one_hot = np_utils.to_categorical(test_y)
    predicted_probs = model.predict(testXCs)
    predicted_classes = np.argmax(predicted_probs, axis=1)
    test_accuracy = sum(predicted_classes == test_y) / test_y.shape[0]
    print("Test set accuracy: {}%".format(test_accuracy))