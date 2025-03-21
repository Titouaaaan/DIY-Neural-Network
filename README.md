# DIY Neural Network Project

## Overview
This project is all about building a neural network from the ground up using only NumPy. No TensorFlow, no PyTorch—just the raw fundamentals of deep learning. 
If you’ve ever wanted to truly understand how neural networks work under the hood, this is pretty helpful.
There's also a bunch of comments that go through the math for forward and backward passes to help explain where the formulas are coming from.
The code base will be extended over time, and yes it's normal that it is slower than using PyTorch.


## Key Features
- **NumPy-based Implementation**: The entire project is built using NumPy, offering a low-level understanding of neural network operations.
- **Modular Design**: The code is organized into modules, making it easy to understand and extend.
- **Variety of Layers and Activations**: Includes implementations of various layers (e.g., Linear, Convolution), activation functions (e.g., Tanh, Sigmoid, ReLU), and loss functions (e.g., MSE, CrossEntropy).
- **Autoencoder and Dimensionality Reduction**: Features implementations of autoencoders and dimensionality reduction techniques like PCA and t-SNE.
- **Training and Optimization**: Provides scripts for training neural networks using stochastic gradient descent (SGD) and other optimization techniques.

## Installation and Usage

### Prerequisites
- Python 3.x
- NumPy
- Matplotlib (for visualization)

### Installation
Clone the repository:
```bash
git clone https://github.com/Titouaaaan/DIY-Neural-Network.git
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project
There are scripts to test out the components of the neural network, which includes training and visualization.
```bash
python -m scripts.SCRIPTNAME
```
for example if you want to test convolution layers
```bash
python -m scripts.conv_networks
```

### Modules
The code is structured in a way that progressively builds up from simple components to more advanced neural network architectures.

- **Activation Functions**: Implementations of common activation functions like Tanh, Sigmoid, and ReLU.
- **Layers**: Basic layer implementations including Linear, Convolution, and Pooling layers.
- **Loss Functions**: Various loss functions such as Mean Squared Error (MSE), CrossEntropy, and Binary CrossEntropy.
- **Sequential Network**: A sequential container that allows stacking layers to form a neural network.
- **Optimizers**: Implementation of Stochastic Gradient Descent (SGD) for training the network.

### Dimensionality Reduction
- **PCA**: Principal Component Analysis for linear dimensionality reduction.
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding for non-linear dimensionality reduction.

### Autoencoder
- **Encoder-Decoder Architecture**: Implementation of an autoencoder for dimensionality reduction and reconstruction tasks.

### Clustering
- **K-Means**: Implementation of the K-Means clustering algorithm.

### Data Handling and Visualization
- **Dataset Generation**: Functions to create and visualize synthetic datasets.
- **MNIST Data Loader**: Utility to load and visualize the MNIST dataset.
- **Visualization Tools**: Functions to plot data, loss curves, and reconstructed images.