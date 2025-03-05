# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.conv_networks
# ===================================================================

import numpy as np
from src.convolution import Convolution1D, MaxPool1D, Flatten
from os.path import join
from src.utils import MnistDataloader, one_hot_encode, plot_loss
from src.encapsulation import Sequential, Optim
from src.activation_functions import ReLU, Softmax
from src.loss import CrossEntropy
from src.linear import Linear

input_path = 'data/MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.array(x_train)
x_test = np.array(x_test)
new_dimension = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(-1, new_dimension).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, new_dimension).astype(np.float32) / 255.0
print(f"x_train shape: {x_train.shape}")
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# Initialize the layers
# Conv1D(3,1,32) → MaxPool1D(2,2) → Flatten() → Linear(416,100) → ReLU() → Linear(100,10)
conv1d = Convolution1D(k_size=3, chan_in=1, chan_out=32, stride=1)
maxpool1d = MaxPool1D(k_size=2, stride=2)
flatten = Flatten()
linear1 = Linear(416, 100)
relu = ReLU()
linear2 = Linear(100, 10)

# Sample input from the MNIST dataset
# Reshape x_train to have a single channel dimension
x_train_reshaped = x_train.reshape(-1, 28, 1)
x_test_reshaped = x_test.reshape(-1, 28, 1)

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

network = Sequential(
    Convolution1D(k_size=3, chan_in=1, chan_out=32, stride=1),
    MaxPool1D(k_size=2, stride=2),
    Convolution1D(k_size=3, chan_in=32, chan_out=64, stride=1),
    MaxPool1D(k_size=2, stride=2),
    Convolution1D(k_size=3, chan_in=64, chan_out=128, stride=1),
    MaxPool1D(k_size=2, stride=2),
    Flatten(),
    Linear(128, 64),  # Adjust input size based on the output of the last conv layer
    ReLU(),
    Linear(64, 32), 
    ReLU(), 
    Linear(32, 10),
    Softmax() 
)

optimizer = Optim(network=network, loss=CrossEntropy(), eps=0.0003)

""" losses = optimizer.SGD(data=(x_train_reshaped, y_train, x_test_reshaped, y_test), batch_size=500, max_epoch=100, test_results=True, verbose=False, multiclass=True)
plot_loss(losses) """

n_samples = 6000
log_loss = []
for i in range(400):
    print(i)
    loss = optimizer.step(batch_x=x_train_reshaped[:n_samples], batch_y=y_train[:n_samples]) 
    log_loss.append(loss)

plot_loss(log_loss)

predictions_probs = network.forward(x_test_reshaped[:n_samples])
predictions = np.argmax(predictions_probs, axis=1)
true_labels = np.argmax(y_test[:n_samples], axis=1)
accuracy = np.sum(predictions == true_labels)
accuracy_percentage = accuracy / len(true_labels) * 100

print(f'Accuracy of model: {accuracy_percentage}%')
 