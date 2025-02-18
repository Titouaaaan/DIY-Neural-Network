from src.module import Module
from src.linear import Linear
from src.activation_functions import TanH, Sigmoid
import numpy as np

class Sequential:
    def __init__(self, *args: Module) -> None:
        self.modules = [*args] # store all our modules in a list

    def forward(self, input):

        for module in self.modules: # iterate through all modules of the network
            input = module.forward(input) # output of a modules forward pass becomes the next one's input

        return input # this will be the final output of the forward pass

    def backward(self, gradient):
        
        for module in reversed(self.modules):
            module.backward_update_gradient(module.last_input, gradient)
            gradient = module.backward_delta(module.last_input, gradient)
        return gradient 
    
    def update_parameters(self, eps):
        for module in self.modules:
            module.update_parameters(learning_rate=eps)
        
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

class Optim:
    def __init__(self, network: Sequential, loss, eps):
        self.network = network
        self.loss = loss
        self.eps = eps 

    def step(self, batch_x, batch_y):
        # feed forward
        output = self.network.forward(batch_x)

        # loss step
        loss = self.loss.forward(batch_y, output)

        # back propagation
        gradient = self.loss.backward(batch_y, output)
        self.network.backward(gradient=gradient)

        # update parameters
        self.network.update_parameters(eps=self.eps)

        return loss
    
    def SGD(self, data: tuple, batch_size: int, max_epoch: int, test_results: bool=True, verbose: bool=True, multiclass: bool=False):
        X_train, y_train, X_test, y_test = data
        n_samples = X_train.shape[0] 
        log_loss = []

        if verbose:
            print(f'We divided our dataset of {n_samples} samples into {n_samples / batch_size} batches')
        
        for epoch in range(max_epoch):
            indices = np.random.permutation(n_samples) # generate a random array of indices

            # shuffle the train dataset
            X_epoch = X_train[indices] 
            y_epoch = y_train[indices]

            # create the mini batches for this epoch
            X_batches = np.array_split(X_epoch, np.ceil(n_samples / batch_size)) # np.ceil rounds up the division, np.ceil(3.125) → 4 gives 4 batches
            y_batches = np.array_split(y_epoch, np.ceil(n_samples / batch_size)) # same for labels

            epoch_loss = [] # log the losses of the epoch
            for batch_X, batch_y in zip(X_batches, y_batches): # iterate through the mini batches
                loss = self.step(batch_X, batch_y) # optim step
                epoch_loss.append(loss) # save each loss from the mini batch
            avg_loss = np.mean(epoch_loss)
            log_loss.append(avg_loss) # save the average loss of all the mini batches as the epoch loss
            if verbose:
                print(f'epoch {epoch}, loss:{np.mean(epoch_loss)}')

        # just test out the model on the test dataset
        if test_results:
            if not multiclass:
                predictions = np.where(self.network.forward(X_test) > 0.5, 1, 0)
                accuracy = np.sum(predictions == y_test)
                print(f'Accuracy of model: {accuracy/len(y_test)*100}%')

            else:
                output = self.network.forward(X_test)
                predictions = np.argmax(output, axis=1)
                y_test_labels = np.argmax(y_test, axis=1)  # If y_test is one-hot
                accuracy = np.mean(predictions == y_test_labels)
                print(f'Accuracy of model: {accuracy * 100}%')

        # return the loss if we want to make graphs
        return log_loss

