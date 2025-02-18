# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.multiclass
# ===================================================================

import random
from os.path  import join
import matplotlib.pyplot as plt
import numpy as np
from src.linear import Linear
from src.loss import MSELoss, CrossEntropy
from src.activation_functions import TanH, Sigmoid, Softmax
from src.encapsulation import Sequential, Optim
from src.utils import MnistDataloader, show_images, plot_loss

# ===================================================================
# LOADING THE DATASET
# the data is already included in the data foldeer (duh)
input_path = 'data/MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Show some random training and test images 
show_images(x_train, y_train, x_test, y_test, n_samples=5)

# ===================================================================
# PRE PROCESSING
# Let's have a look at our dataset first to see what we're working with
# a list of n elements (60k for training)
# each element (picture) is a list of size 28
# the list contains 28 np arrays of size 28, -> 1D array of length 28
# so we have a dataset of pictures of size 28x28
print(f"x_train length: {len(x_train)}, y_train length: {len(y_train)}")
print(f"x_test length: {len(x_test)}, y_test length: {len(y_test)}")
for i in range(1):
    elem = x_train[i]
    print(f'Elem type: {type(elem)} of length {len(elem)}, which contains a {type(elem[0])} of shape {elem[0].shape}')

# Problem is that our network doesn't take lists as input which means we need to convert this into an array:
x_train = np.array(x_train)
x_test = np.array(x_test)

# Check the shapes
print('\nX shape before pre processing:')
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# other problem: our linear module takes as input something of the shape (batch_size, input_dim)
# But right now we have something of the shape (batch_size, input_dim_i, input_dim_y) which won't work
# so we need to flatten the dimension
new_dimension = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(-1, new_dimension)
x_test = x_test.reshape(-1, new_dimension)

# normalize data to have values between 0 and 1 instead of 0 and 255
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Check the shapes after flattening
print('\nAfter the reshape:')
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"Sample from dataset: {x_train[0]}")

# Careful, we need to do different pre processing step for the labels 
# first, convert them into an array since they are currently lists
# then one hot encode them (otherwise we will have array shape mismatches)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Check the shape after conversion
print('\nConverting the labels into arrays')
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f'First 5 label: {y_train[:5]}')

# function to one hot encode the labels
def one_hot_encode(y, num_classes=10):
    # Create a zero matrix of shape (num_samples, num_classes)
    one_hot = np.zeros((y.shape[0], num_classes))
    # Set the correct class as 1 for each sample
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

print('\nAfter one-hot encoding:')
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f'First 5 label: {y_train[:5]}\n')

# ===================================================================
# CREATING THE NETWORK
# Let's create our network for the multi class problem:
input_dim = x_train.shape[1]
output_dim = 10
learning_r = 0.001
epoch = 150
log_loss = []
n_samples = 256

network = Sequential(
    Linear(input_dim, 128),   # First hidden layer with 128 units
    TanH(),                   # Activation function
    Linear(128, output_dim),   # Output layer with 10 units (one for each class)
    Softmax()                 # Softmax for output probabilities
)

optimizer = Optim(network=network, loss=CrossEntropy(), eps=learning_r)

losses = optimizer.SGD(data=(x_train, y_train, x_test, y_test), batch_size=n_samples, max_epoch=epoch, test_results=True, verbose=True, multiclass=True)
plot_loss(losses)