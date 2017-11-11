import sys
sys.path.insert(0, '../')

import pdb
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_raw_data
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

from Conv_NN_layer_utils import *

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_small_data(data, num_train):
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
    return small_data

def plot_train_history(solver):
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


def main():
    # Load the (preprocessed) CIFAR10 data.
    data = get_CIFAR10_raw_data()
    for k, v in data.iteritems():
        print '%s: ' % k, v.shape

    # Get small data for finetuning
    small_data = get_small_data(data, 5000)
    
    # Network Architecture
    # {conv- [batch norm] - relu - pool}
    cnn_layer_1 = (64, 3, 1, 1)
    pool_layer_1 = (2, 2, 2)
    layer_1 = (cnn_layer_1, pool_layer_1)
    cnn_layer_2 = (128, 3, 1, 1)
    pool_layer_2 = (2, 2, 2)
    layer_2 = (cnn_layer_2, pool_layer_2)
    cnn_layer_3 = (256, 3, 1, 1)
    pool_layer_3 = (2, 2, 2)
    layer_3 = (cnn_layer_3, pool_layer_3)
    hidden_dims_CNN = (layer_1, layer_2, layer_3)

    # {affine - [batch norm] - relu - [dropout]}
    fc_layer_1 = 256
    drop_layer_1 = 1
    layer_1 = (fc_layer_1, drop_layer_1)
    fc_layer_2 = 128
    drop_layer_2 = 1
    layer_2 = (fc_layer_2, drop_layer_2)
    hidden_dims_FC = (layer_1, layer_2)

    num_classes = 10
    
    model = ConvNet( input_dim=(3, 32, 32), hidden_dims_CNN = hidden_dims_CNN,
                    hidden_dims_FC = hidden_dims_FC, num_classes = num_classes, 
                    weight_scale=1e-2, reg=0.001, dtype=np.float32)

    select_num_train_data = 0
    test_weght_scale = 0
    test_lr = 1

    # Test how many data is enough for training
    if select_num_train_data == 1:
        num_train = (500, 1000, 5000, 10000)
        epoch = (20, 10, 2, 1)
        for i in range(0, len(num_train)):
            print 'num_train_data : %d' % (num_train[i])

            small_data = get_small_data(data, num_train[i])
            solver = Solver(model, small_data,
                        num_epochs=epoch[i], batch_size=100,
                        update_rule='sgd_momentum',
                        optim_config={
                            'learning_rate': 1e-3,
                        },
                        verbose=False, print_every=20)
        
            solver.train()
            print 'num_train : %d, train_acc : %f, val_acc : %f' % (num_train[i], 
                                                                    solver.train_acc_history[-1],
                                                                    solver.val_acc_history[-1])


    # Test settings of weight initialization 
    if test_weght_scale == 1:
        weight_scale = (1e-2, 1e-3, -1)
        for i in range(0, len(weight_scale)):
            print 'weight_scale : %f' % (weight_scale[i])
            model = ConvNet( input_dim=(3, 32, 32), hidden_dims_CNN = hidden_dims_CNN,
                        hidden_dims_FC = hidden_dims_FC, num_classes = num_classes, 
                        weight_scale=weight_scale[i], reg=0.001, dtype=np.float32)

            solver = Solver(model, small_data,
                        num_epochs=2, batch_size=100,
                        update_rule='sgd_momentum',
                        optim_config={
                            'learning_rate': 1e-3,
                        },
                        verbose=True, print_every=20)
            solver.train()

            print 'weight_scale : %f, train_acc : %f, val_acc : %f' % (weight_scale[i], 
                                                                    solver.train_acc_history[-1],
                                                                    solver.val_acc_history[-1])
    if test_lr == 1:
        lr = (1e-2, 1e-3, 1e-4)
        for i in range(0, len(lr)):
            print 'lr : %f' % (lr[i])
            model = ConvNet( input_dim=(3, 32, 32), hidden_dims_CNN = hidden_dims_CNN,
                        hidden_dims_FC = hidden_dims_FC, num_classes = num_classes, 
                        weight_scale=-1, reg=0.001, dtype=np.float32)

            solver = Solver(model, small_data,
                        num_epochs=10, batch_size=100,
                        update_rule='sgd_momentum',
                        optim_config={
                            'learning_rate': lr[i],
                        },
                        verbose=True, print_every=10)
            solver.train()

            print 'lr : %f, train_acc : %f, val_acc : %f' % (lr[i], 
                                                            solver.train_acc_history[-1],
                                                            solver.val_acc_history[-1])

if __name__ == '__main__':
    main()