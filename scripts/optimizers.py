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
from src.utils import create_moon_dataset, train_test_split, plot_train_test_split, plot_loss
import matplotlib.pyplot as plt
import numpy as np

# Time to encapsulate the Network, to avoid having to do a for loop for training (which would get bigger as the network size increases), 
# and also to avoid having to do this 
# ```test_prediction = activation_fun_2.forward(layer2.forward(activation_fun_1.forward(layer1.forward(X_test))))``` when trying to predict labels!

# We now have a Sequential class which let's us group the different elements of the network together! We can now create our network like this:
input_dim = 2
output_dim = 1
learning_r = 0.1
epoch = 2000
log_loss = []
# From my tests we usually get an accuracy of 84 to 88% (depending on your parameter choice), 
# So let's see if we can recreate those results

# recreate the moon dataset but with different params this time!
n_samples = 2000
theta = np.linspace(0, np.pi, n_samples)
X, y = create_moon_dataset(n_samples=n_samples, noise=0.2, theta=theta)
X_train, y_train, X_test, y_test = train_test_split(X=X, y=y, split_ratio=0.8)

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