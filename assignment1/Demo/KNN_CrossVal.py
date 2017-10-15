import random
import pdb
import sys
sys.path.insert(0, '../')

import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10 
from cs231n.classifiers import KNearestNeighbor

cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

Train_Mix = np.hstack([X_train, y_train])
train_folds = np.array_split(Train_Mix, 5, axis = 0)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.zeros((5,train_folds[0].shape[0], train_folds[0].shape[1]-1))
y_train_folds = np.zeros((5,train_folds[0].shape[0], 1))
for i in range(0, num_folds):
    X_train_folds[i,:,:] = train_folds[i][:,0:-1]
    y_train_folds[i,:,:] = train_folds[i][:,-1].reshape(train_folds[i].shape[0],1)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}
classifier = KNearestNeighbor()

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
"""

fold_index = np.array(range(num_folds))
for i in range(0, len(k_choices)):
    k = k_choices[i]
    print(k)
    accuracy = 0
    for test_fold in range(0, num_folds):
        X_test_ = X_train_folds[test_fold,:,:]
        y_test_ = y_train_folds[test_fold,:,:]
        train_idx = (fold_index != test_fold)
        X_train_ = X_train_folds[train_idx,:,:]
        X_train_ = X_train_.reshape(X_train_.shape[0]*X_train_.shape[1], X_train_.shape[2])
        y_train_ = y_train_folds[train_idx,:,:]
        y_train_ = y_train_.reshape(y_train_.shape[0]*y_train_.shape[1], y_train_.shape[2])

        classifier.train(X_train_, y_train_)

        num_test = X_test_.shape[0]
        dists = classifier.compute_distances_no_loops(X_test_)
        y_test_pred = classifier.predict_labels(dists, k)
        num_correct = np.sum(y_test_pred[:] == y_test_[:,0])
        accuracy = accuracy + float(num_correct) / num_test
    accuracy = accuracy / num_folds
    k_to_accuracies[k] = accuracy
    print(accuracy)
pdb.set_trace()
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)

# plot the raw observations
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

"""

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 1

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
#y_test_pred = classifier.predict(X_test, k=best_k)
dists_two = classifier.compute_distances_no_loops(X_test)
y_test_pred = classifier.predict_labels(dists_two, k=best_k)
# Compute and display the accuracy
num_test = y_test.shape[0]
num_correct = np.sum(y_test_pred[:] == y_test[:,0])
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)