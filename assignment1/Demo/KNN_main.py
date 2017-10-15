import random
import pdb
import sys
sys.path.insert(0, '../')

import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10 
from cs231n.classifiers import KNearestNeighbor

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        #plt.imshow(X_train[idx].astype(np.uint8))
        if i == 0:
            plt.title(cls)
#plt.show()

num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

#dists = classifier.compute_distances_two_loops(X_test)
#dists_one = classifier.compute_distances_one_loop(X_test)
dists_two = classifier.compute_distances_no_loops(X_test)

y_test_pred = classifier.predict_labels(dists_two, 1)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
#difference_one = np.linalg.norm(dists - dists_one, ord='fro')
#print 'Difference was: %f' % (difference_one, )
#if difference_one < 0.001:
#  print 'Good! The distance matrices are the same'
#else:
#  print 'Uh-oh! The distance matrices are different'

#difference_two = np.linalg.norm(dists - dists_two, ord='fro')
#print 'Difference was: %f' % (difference_two, )
#if difference_two < 0.001:
#  print 'Good! The distance matrices are the same'
#else:
#  print 'Uh-oh! The distance matrices are different'
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

#two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
#print 'Two loop version took %f seconds' % two_loop_time

#one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
#print 'One loop version took %f seconds' % one_loop_time

#no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
#print 'No loop version took %f seconds' % no_loop_time

pdb.set_trace()