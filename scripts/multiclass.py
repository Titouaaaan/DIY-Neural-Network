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
from src.utils import MnistDataloader


# MAKE SURE YOU DOWNLOAD THE DATASET YOURSELF: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data
# also pay attention to the paths names and structure
# I didn't add the data to the repo because it takes a lot of space (but it doesn't take long to download)
# change the input_path if you stored the data somewhere else
input_path = 'data/MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Show some random training and test images 
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

print(f"x_train shape: {len(x_train)}, y_train shape: {len(y_train)}")
print(f"x_test shape: {len(x_test)}, y_test shape: {len(y_test)}")
n_samples = 5
fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
for i in range(n_samples):
    axes[i].imshow(x_train[i], cmap=plt.cm.gray)
    axes[i].set_title(f"label {i+1}: {y_train[i]}")
    axes[i].axis('off')  # Hide axes
plt.tight_layout()
plt.show()

