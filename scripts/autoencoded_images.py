# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.autoencoded_images
# ===================================================================

import random
from os.path  import join
import matplotlib.pyplot as plt
import numpy as np
from src.linear import Linear
from src.auto_encoder import Autoencoder
from src.loss import BinaryCrossEntropy
from src.activation_functions import TanH, Sigmoid
from src.encapsulation import Sequential
from src.utils import MnistDataloader, show_images, plot_loss

# Time to reuse the MNIST data set from the multiclass script for the autoencoder!
input_path = 'data/MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Show some random training and test images 
show_images(x_train, y_train, x_test, y_test, n_samples=5)

# Time to redo some pre processing for the images
# Convert to array -> reshape to a flat dimension -> normalize
x_train = np.array(x_train)
x_test = np.array(x_test)
new_dimension = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(-1, new_dimension).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, new_dimension).astype(np.float32) / 255.0
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# Architecture example for an auto-encoder:
# Encoder : Linear(784 ,256) → TanH() → Linear(256,100) → TanH() → Linear(100,32) → TanH()
# Decoder : Linear(32 ,100) → TanH() → Linear(100 , 256) → TanH() → Linear(256 , 784) → Sigmoide()
input_dim = x_train.shape[1]
output_dim = 32
learning_rate = 0.01
max_epoch = 200
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

auto_encoder = Autoencoder(encoder=encoder, decoder=decoder, loss=BinaryCrossEntropy(), learning_rate=learning_rate)
losses = auto_encoder.train_auto_encoder(data=(x_train, x_test), batch_size=n_samples, max_epoch=max_epoch)
plot_loss(log_loss=losses)




