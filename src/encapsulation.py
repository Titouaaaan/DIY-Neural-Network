from src.module import Module
from src.linear import Linear
from src.activation_functions import TanH, Sigmoid

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