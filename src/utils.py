import numpy as np
import matplotlib.pyplot as plt
import struct
from array import array
import random

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

#
# MNIST Data Loader Class
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  

def show_images(x_train, y_train, x_test, y_test, n_samples):
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    _ , axes = plt.subplots(1, n_samples, figsize=(15, 3))
    for i in range(n_samples):
        axes[i].imshow(x_train[i], cmap=plt.cm.gray)
        axes[i].set_title(f"label {i+1}: {y_train[i]}")
        axes[i].axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()