from typing import Any

class Loss(object):
    def forward(self, y, yhat):
        '''
        calculate cost based on the two inputs y and yhat
        '''
        raise NotImplementedError()
    
    def backward(self, y, yhat):
        '''
        calculate the gradient of the cost in relation to yhat 
        '''
        raise NotImplementedError()

class Module(object):
    def __init__(self):
        self._parameters = None # parameters of the module (like the matrix of weights for a linear module)
        self._gradient = None # accumulate the passed gradients

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def zero_grad(self):
        '''
        reset gradient to 0
        '''
        raise NotImplementedError()

    def forward(self, X):
        ''' 
        calculate outputs of a module based on the data X passed as input
        '''
        raise NotImplementedError()

    def update_parameters(self, learning_rate=1e-3):
        '''
        update the parameters of the module based on the accumulated gradient
        until the next call, with a gradient step
        '''
        raise NotImplementedError()

    def backward_update_gradient(self, input, delta):
        '''
        calculate the gradient of the cost in relation to the parameters of the module
        and add it to the _gradient variable based on the input and the delta (gradient) of the next layer
        '''
        raise NotImplementedError()

    def backward_delta(self, input, delta):
        '''
        calculate the gradient of the cost in relation to the input of the deltas of the next layer
        '''
        raise NotImplementedError()
    
    def reset_parameters(self):
        '''
        reset the weights of the module
        '''
        raise NotImplementedError()