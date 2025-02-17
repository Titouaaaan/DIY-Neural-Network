# ===================================================================
# USE THIS TO RUN THE SCRIPTS!
# python -m scripts.stochastic_GD
# ===================================================================

from src.linear import Linear
from src.loss import MSELoss
from src.activation_functions import TanH, Sigmoid
from src.encapsulation import Sequential, Optim
from src.utils import create_moon_dataset, train_test_split, plot_train_test_split, plot_loss
import matplotlib.pyplot as plt
import numpy as np

# ===================================================================
# Okay let's implement SGD: stochastic gradient descent
# To update the model's parameters we compute the gradient of the loss function using 
# randomly selected small batches per iteration, rather than the entire dataset 
# This makes it faster and more scalable for large datasets but introduces more variance in updates, 
# which can help escape local minima and improve generalization.
# ===================================================================

# As usual we set the parameters:
input_dim = 2
output_dim = 1
learning_r = 0.06
epoch = 2000
log_loss = []
n_samples = 2000

# create the network
network = Sequential(
    Linear(input_dim, 6),
    TanH(),
    Linear(6, output_dim),
    Sigmoid(),
    )

# create the optimizer
optimizer = Optim(network=network, loss=MSELoss(), eps=learning_r)

# create the dataset
X, y = create_moon_dataset(n_samples=n_samples, noise=0.2, theta=np.linspace(0, np.pi, n_samples))

# run the SGD function of the optmizer
# set test_results to true to test the results and verbose to true to display information
# go read src/encapsulation.py for more info on this function !
losses = optimizer.SGD(train_test_split(X, y, split_ratio=0.8), batch_size=200, max_epoch=200, test_results=True, verbose=False)
plot_loss(losses)