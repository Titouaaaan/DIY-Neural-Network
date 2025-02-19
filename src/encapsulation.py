from src.module import Module, Loss
import numpy as np

class Sequential:
    '''
    This class will essentially be the network, which contains 
    a sequences of various modules one after the other, 
    like linear layers and activation functions

    Example use case:
    network = Sequential(
        Linear(input_dim, 128),   # First hidden layer with 128 units
        TanH(),                   # Activation function
        Linear(128, output_dim),   # Output layer 
        Softmax()                 # Softmax for output probabilities
    )
    '''
    def __init__(self, *args: Module) -> None:
        self.modules = [*args] # store all our modules in a list

    def forward(self, input):
        '''
        Call the forward function of each module (basic forward pass)
        '''
        for module in self.modules: # iterate through all modules of the network
            input = module.forward(input) # output of a modules forward pass becomes the next one's input

        return input # this will be the final output of the forward pass

    def backward(self, gradient):
        ''' 
        Same as above but for backpropagation :)
        '''
        for module in reversed(self.modules):
            module.backward_update_gradient(module.last_input, gradient)
            gradient = module.backward_delta(module.last_input, gradient)
        return gradient 
    
    def update_parameters(self, eps: float):
        '''
        Again we just call the module's function
        But some modules dont update parameters, like activation functions
        So we have to make sure that their call doesn't break
        -> have pass in the update_parameters fun of the activation function
        '''
        for module in self.modules:
            module.update_parameters(learning_rate=eps)
        
    def zero_grad(self):
        '''
        call zero grad where possible,
        same logic as for the function above
        '''
        for module in self.modules:
            module.zero_grad()

class Optim:
    '''
    Optimizer class for training a neural network using stochastic gradient descent (SGD
    '''
    def __init__(self, network: Sequential, loss: Loss, eps: float):
        '''
        Initialize the optimizer with a:
        - network: model to train
        - loss: loss function 
        - eps: learning rate for updating the network parameters
        '''
        self.network = network
        self.loss = loss
        self.eps = eps 

    def step(self, batch_x, batch_y):
        '''
        Perform a single optimization step (forward pass, loss computation, backward pass, and parameter update).
        '''
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
        '''
        Train the neural network using stochastic gradient descent (SGD).

        Parameters:
        - data: tuple containing training and test datasets (X_train, y_train, X_test, y_test)
        - batch_size: trivial
        - max_epoch: max training steps
        - test_results: whether to test the model on the test dataset after training
        - verbose: for printing statements
        - multiclass: multiclass classification problem or not?

        Returns:
        - log_loss: A list of average losses for each epoch.

        The test results at the bottom is subject to change but the rest should work consistently
        '''
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