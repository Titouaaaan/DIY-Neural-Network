# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.optimizers
# ===================================================================
# To learn better play around with the various parameters, 
# like the amount of samples, the learning rate, the noise etc...
# ===================================================================

from src.linear import Linear
from src.loss import MSELoss
from src.activation_functions import TanH, Sigmoid
from src.encapsulation import Sequential, Optim
import matplotlib.pyplot as plt
import numpy as np

# dataset generation
def create_moon_dataset(n_samples: int, noise: float, theta: np.array):
    # Class 0: Upper moon
    x0 = np.cos(theta) + np.random.normal(0, noise, n_samples) # take cosine of theta for x coordinates, and add noise for a nicer distribution
    y0 = np.sin(theta) + np.random.normal(0, noise, n_samples) # same but take the sine for the y coordinates (+ noise)
    labels0 = np.zeros(n_samples) # create n sample 'zero labels'

    # Class 1: Lower moon (shifted and rotated)
    x1 = 1 - np.cos(theta) + np.random.normal(0, noise, n_samples) # same logic as before but shifts to the right by 1 unit
    y1 = -np.sin(theta) + 0.5 + np.random.normal(0, noise, n_samples) # taking -sin(theta) flips the moon, we add the 0.5 to shift it downward and add noise
    labels1 = np.ones(n_samples) # create n sample 'one labels'

    # Combine data
    # for X, first n_samples belong to class 0, rest 1
    X = np.vstack([np.column_stack((x0, y0)), np.column_stack((x1, y1))]) # create the dataset by combining the 0 and 1 class
    y = np.concatenate([labels0, labels1])
    
    return X, y

# plot dataset of two classes
def plot_data(X: np.array, y: np.array, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# split the dataset according to a ratio (80%, 75% etc...)
def train_test_split(X: np.array, y: np.array, split_ratio: float):
    indices = np.random.permutation(len(X)) # random permutation of indices from 0 to len(X)-1
    # print(f'sample of the new order of indices: {indices[:10]}')

    # shuffle X and y using the same permutation to avoid mixing their labels
    # random permutation, but same for both arrays!
    # ex: the index 0 is now randomly (for this example) at index 156 for both arrays
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # we want 80% of train
    # calculate the index of where the split should happen
    # for example with n sample = 2000, 2000 * 0.8 = 1600
    split_index = int(len(X_shuffled) * split_ratio) 

    # Split the data
    X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:] # splice the X_shuffled array into two new arrays, split at index 'split_index'
    y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:] # same for y_shuffled

    #add extra column for matrix sizes to be compatible with rest of the code, so (1600,) -> (1600,1)
    y_train = y_shuffled[:split_index].reshape(-1, 1)  
    y_test = y_shuffled[split_index:].reshape(-1, 1)

    return X_train, y_train, X_test, y_test

# plot the train and test dataset into two seperate graphs next to each other (still 2 classes only)
def plot_train_test_split(X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
    # Plot training and test data with reshaped labels
    plt.figure(figsize=(12, 5))

    # Training data (flatten y_train to 1D for indexing)
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[y_train.ravel()==0, 0], X_train[y_train.ravel()==0, 1], c='blue', label='Class 0 (Train)')
    plt.scatter(X_train[y_train.ravel()==1, 0], X_train[y_train.ravel()==1, 1], c='red', label='Class 1 (Train)')
    plt.title("Training Data")
    plt.legend()

    # Test data (flatten y_test to 1D for indexing)
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[y_test.ravel()==0, 0], X_test[y_test.ravel()==0, 1], c='blue', marker='o', alpha=0.6, label='Class 0 (Test)')
    plt.scatter(X_test[y_test.ravel()==1, 0], X_test[y_test.ravel()==1, 1], c='red', marker='o', alpha=0.6, label='Class 1 (Test)')
    plt.title("Test Data")

    plt.tight_layout()
    plt.show()

# plot the loss over time
def plot_loss(log_loss: list):
    plt.figure(figsize=(10, 6))
    plt.plot(log_loss, label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('loss over time')
    plt.legend()
    plt.show()

# Time to encapsulate the Network, to avoid having to do a for loop for training (which would get bigger as the network size increases), 
# and also to avoid having to do this 
# ```test_prediction = activation_fun_2.forward(layer2.forward(activation_fun_1.forward(layer1.forward(X_test))))``` when trying to predict labels!

# We now have a Sequential class which let's us group the different elements of the network together! We can now create our network like this:
input_dim = 2
output_dim = 1
learning_r = 0.1
epoch = 4000
log_loss = []

# recreate the moon dataset but with different params this time!
n_samples = 2000
theta = np.linspace(0, np.pi, n_samples)
X, y = create_moon_dataset(n_samples=n_samples, noise=0.2, theta=theta)
X_train, y_train, X_test, y_test = train_test_split(X=X, y=y, split_ratio=0.8)

# parameters
input_dim = 2 # input of 2 because we have 2 coordinates for each point
output_dim = 1 # 1 because we predict one class, 1 or 0
learning_r = 0.1 # play around with that, but a value around 0.01 is pretty good
num_epoch = 2000 # could even lower that a bit based on results but i gets executed very fast anyways

# basic visualisation
plot_train_test_split(X_train, y_train, X_test, y_test)

# creating the Sequential network,
# Here it is composed of two linear layers and two activation functions 
network = Sequential(
    Linear(input_dim, 6),
    TanH(),
    Linear(6, output_dim),
    Sigmoid(),
    )

# creating the optimizer, here we use the MSE loss
optimizer = Optim(network=network, loss=MSELoss(), eps=learning_r)

# training (optimizing) loop of n epochs
for epoch in range(epoch):
    loss = optimizer.step(batch_x=X_train, batch_y=y_train)  
    log_loss.append(loss)

# let's see if the loss looks good (should decrease relatively quickly and then stabilize/converge to a value)
plot_loss(log_loss=log_loss)

#calculation of the accuracy
predictions = np.where(network.forward(X_test) > 0.5, 1, 0)
accuracy = np.sum(predictions == y_test)
print(f'Accuracy of model: {accuracy/len(y_test)*100}%') 