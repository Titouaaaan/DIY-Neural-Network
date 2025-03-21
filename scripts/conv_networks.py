# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.conv_networks
# ===================================================================

import numpy as np
from os.path import join
from src.convolution import Convolution1D, Flatten
from src.pooling import MaxPool1D, AveragePool1D
from src.utils import MnistDataloader, one_hot_encode, plot_loss
from src.encapsulation import Sequential, Optim
from src.activation_functions import ReLU, Softmax
from src.loss import CategoricalCrossEntropy
from src.linear import Linear

"""
This script trains a convolutional neural network (CNN) on the MNIST dataset, which is a collection of handwritten digit images. 
The goal is to classify each image into one of ten categories, representing the digits 0 through 9. 
We start by loading and preprocessing the data, normalizing the images, and one-hot encoding the labels. 
The CNN architecture includes two convolutional layers with max-pooling, followed by a flattening layer and two fully connected layers with ReLU activation. 
Concerning the architecture, play around with that and see what works well, what i did here is probably not the most optimized network!
The network is trained using SGD. 
The script also includes sections for testing individual layers and visualizing intermediate outputs (all the way at the bottom), 
which can be useful for debugging and understanding the model's behavior. 
Adjust the number of training samples to balance between training speed and accuracy, 
and ensure that input dimensions match the expected dimensions at each layer to avoid shape mismatch errors.
"""


input_path = 'data/MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Quick reshaping and pre-processing for the data to be usuable
x_train = np.array(x_train) # transform data into arrays
x_test = np.array(x_test) # same
new_dimension = x_train.shape[1] * x_train.shape[2] # flattened dimension of the image, in the this its 28 * 28
x_train = x_train.reshape(-1, new_dimension).astype(np.float32) / 255.0 # normalize values
x_test = x_test.reshape(-1, new_dimension).astype(np.float32) / 255.0
print(f"x_train shape: {x_train.shape}")

# same for the labels
y_train = np.array(y_train) # convert labels to arrays
y_test = np.array(y_test) # same
y_train = one_hot_encode(y_train) # one hot encode, see function in utils.py
y_test = one_hot_encode(y_test)

# Sample input from the MNIST dataset
# Reshape x_train to have a single channel dimension
x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test_reshaped = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(f"x_train_reshaped shape: {x_train_reshaped.shape}")
print(f"x_test_reshaped shape: {x_test_reshaped.shape}")

# Basic architecture
# Play around with this it's pretty important
network = Sequential(
    Convolution1D(k_size=3, chan_in=1, chan_out=16, stride=2),  
    MaxPool1D(k_size=2, stride=2),
    Convolution1D(k_size=3, chan_in=16, chan_out=32, stride=2),  
    MaxPool1D(k_size=2, stride=2),
    Flatten(),
    Linear(1536, 100),  # Adjust input size based on the new output of the last conv layer (run and see what shape mismatch you get)
    ReLU(),
    Linear(100, 10),
    Softmax()
)
# From what ive test, softmax at the end helps quite a bit for exmaple
# change the amount of conv layers and see what happens (in terms of time too)

n_samples = 10000 # change that based on how fast you want the training to be (but also you lose on accuracy)

# For the optmizer and sgd its the same as the previous scripts 
# so if you want a more detailed explanation go check out optimizers.py (and stochastic_GD.py)
# also the learning rate eps plays a very important role 
# so play around with it to see how it changes the learning process (something between 0.0001 and 0.0005 should do the trick)
optimizer = Optim(network=network, loss=CategoricalCrossEntropy(), eps=0.0003)

# Make sure to leave test_results as True to test the model (and verbose if you want to see the speed of the network)
losses = optimizer.SGD(
    data=(x_train_reshaped[:n_samples], y_train[:n_samples], x_test_reshaped, y_test), 
    batch_size=256, max_epoch=20, test_results=True, verbose=True, multiclass=True)
plot_loss(losses) 

# ==================================================================================================
# Below was just a test to check that everything works, 
# you can uncomment it to test what the outputs and inputs look like
# Might be outdated idk
# But its nice to see what data is being passed around
""" 
# Initialize the layers
# Conv1D(3,1,32) → MaxPool1D(2,2) → Flatten() → Linear(416,100) → ReLU() → Linear(100,10)
conv1d = Convolution1D(k_size=3, chan_in=1, chan_out=32, stride=1)
maxpool1d = MaxPool1D(k_size=2, stride=2)
flatten = Flatten()
linear1 = Linear(416, 100)
relu = ReLU()
linear2 = Linear(100, 10)

# Select a single sample for demonstration
sample_input = x_train_reshaped[:10]  # Shape: (1, 28, 1)

# Forward pass
conv_output = conv1d.forward(sample_input)
pool_output = maxpool1d.forward(conv_output)
flat_output = flatten.forward(pool_output)
linear1_output = linear1.forward(flat_output)
relu_output = relu.forward(linear1_output)
linear2_output = linear2.forward(relu_output)

print("Conv1D Output Shape:", conv_output.shape)
print("MaxPool1D Output Shape:", pool_output.shape)
print("Flatten Output Shape:", flat_output.shape)
print("Linear1 Output Shape:", linear1_output.shape)
print("ReLU Output Shape:", relu_output.shape)
print("Linear2 Output Shape:", linear2_output.shape)

# Dummy gradient for the purpose of demonstration
dummy_gradient = np.random.randn(*linear2_output.shape)

# Backward pass
linear2_delta = linear2.backward_delta(relu_output, dummy_gradient)
relu_delta = relu.backward_delta(linear1_output, linear2_delta)
linear1_delta = linear1.backward_delta(flat_output, relu_delta)
flatten_delta = flatten.backward_delta(pool_output, linear1_delta)
pool_delta = maxpool1d.backward_delta(conv_output, flatten_delta)
conv_delta = conv1d.backward_delta(sample_input, pool_delta)

# Update parameters
conv1d.backward_update_gradient(sample_input, pool_delta)
conv1d.update_parameters(learning_rate=0.01)
linear1.backward_update_gradient(flat_output, relu_delta)
linear1.update_parameters(learning_rate=0.01)
linear2.backward_update_gradient(relu_output, dummy_gradient)
linear2.update_parameters(learning_rate=0.01)

print(linear2.forward(relu_output))
print(linear2.forward(relu_output).shape)
print("Backward pass completed.") 
"""