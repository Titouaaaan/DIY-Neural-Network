# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.autoencoder_high_compression
# 
# THIS SCRIPTS IS NOT FINISHED
# ===================================================================

from os.path  import join
import numpy as np
from src.linear import Linear
from src.auto_encoder import Autoencoder
from src.loss import BinaryCrossEntropy
from src.activation_functions import TanH, Sigmoid
from src.encapsulation import Sequential
from src.utils import MnistDataloader, plot_loss, show_images_side_by_side
from src.dim_reduction import TSNE, PCA
from src.k_means import KMeans  

input_path = 'data/MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

def reshape_data(x_train, x_test):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    new_dimension = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(-1, new_dimension).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, new_dimension).astype(np.float32) / 255.0
    return x_train, x_test

# ===================================================================
# Experiment 1: Autoencoder with high compression
# ===================================================================

# We will use a 3 layer autoencoder with 256, 100 and 10 neurons in each layer
# Since the ouput layer is of size 10, we get a very high compression of the data

x_train, x_test = reshape_data(x_train=x_train,x_test=x_test)
input_dim = x_train.shape[1] # 784 because 28*28
output_dim = 16 # play around with this parameter to observe better results
learning_rate = 0.01
max_epoch = 80
n_samples = 256

encoder = Sequential(
    Linear(input_dim=input_dim, output_dim=256),
    TanH(),
    Linear(input_dim=256, output_dim=100),
    TanH(),
    Linear(input_dim=100, output_dim=output_dim),
    TanH()
)

decoder = Sequential(
    Linear(input_dim=output_dim, output_dim=100),
    TanH(),
    Linear(input_dim=100, output_dim=256),
    TanH(),
    Linear(input_dim=256, output_dim=input_dim),
    Sigmoid()
)

auto_encoder = Autoencoder(encoder=encoder, 
                           decoder=decoder, 
                           loss=BinaryCrossEntropy(), 
                           learning_rate=learning_rate)

losses = auto_encoder.train_auto_encoder(data=(x_train, x_test), 
                                         batch_size=n_samples, 
                                         max_epoch=max_epoch,
                                         show_results=False) # avoid showing results here we'll do it after

plot_loss(log_loss=losses)

# Example usage
# Assuming x_test and reconstructed_images are your image arrays
reconstructed_images = auto_encoder.forward(x_test[:10]) # forward pass on the test sets
show_images_side_by_side(x_test[:10], reconstructed_images, title='Original vs Reconstructed Images')

# ===================================================================
# feed forward on the whole dataset, and use k means clustering and PCA to visualize the data

reconstructed_images = auto_encoder.forward(x_test)
y_test = np.array(y_test)

# How to know if we should use T-SNE or PCA?
# https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/

# Apply PCA to reduce to 2D
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(reconstructed_images)
tsne = TSNE(n_components=2, perplexity=30, learning_rate=50, n_iter=800)
reduced_data = tsne.fit_transform(reconstructed_images)

# Perform K-means clustering on the 2D data
kmeans = KMeans(k=10, points=reduced_data[:750])  # only apply tsne on less samples instead of 10k otherwise it takes too long
cluster_labels = kmeans.run_clustering(y_test[:750])
