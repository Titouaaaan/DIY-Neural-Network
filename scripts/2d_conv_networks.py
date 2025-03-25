# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.2d_conv_networks
# ===================================================================

import numpy as np
from os.path import join
from src.convolution import Flatten, Convolution2D
from src.pooling import MaxPool2D
from src.utils import MnistDataloader, one_hot_encode, plot_loss
from src.encapsulation import Sequential, Optim
from src.activation_functions import ReLU, Softmax
from src.loss import CategoricalCrossEntropy
from src.linear import Linear

"""
Please go check out the conv_networks script first before this one since 
it has more comments explaining what we're doing here!
This script is mostly just to test that the 2D conv layer works as intended
"""


input_path = 'data/MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.array(x_train) # transform data into arrays
x_test = np.array(x_test) # same
x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1).astype(np.float32) / 255.0 # normalize values
x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1).astype(np.float32) / 255.0

y_train = np.array(y_train) # convert labels to arrays
y_test = np.array(y_test) # same
y_train = one_hot_encode(y_train) # one hot encode, see function in utils.py
y_test = one_hot_encode(y_test)

network = Sequential(
    Convolution2D(k_size=3, chan_in=1, chan_out=16, stride=1),  
    ReLU(),                                                    
    MaxPool2D(k_size=2, stride=2),                             
    Flatten(),                                                 
    Linear(input_dim=16*13*13, output_dim=128),                
    ReLU(),                                                   
    Linear(input_dim=128, output_dim=10),                      
    Softmax()
)

n_samples = 200
optimizer = Optim(network=network, loss=CategoricalCrossEntropy(), eps=0.0003)
losses = optimizer.SGD(
    data=(x_train[:n_samples], y_train[:n_samples], x_test, y_test), 
    batch_size=20, max_epoch=15, test_results=True, verbose=True, multiclass=True)
plot_loss(losses) 