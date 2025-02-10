import numpy as np
from src.module import Module

class TanH(Module):
    '''
    z = input
    tanh(z) = (e^z - e^-z) / (e^z + e^-z)
    '''
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        ''' no gradient to reset to zero '''
        pass

    def forward(self, X):
        # print((np.tanh(X)).shape)
        self.last_input = X # keep track of the last input of the module
        return np.tanh(X) # or (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X)) but the np function is faster so let's use that
    
    def update_parameters(self, learning_rate):
        ''' no parameters to update '''
        pass

    def backward_update_gradient(self, input, delta):
        ''' no gradient to update '''
        pass

    def backward_delta(self, input, delta):
        '''
        ∂tanh = 1 - z**2

        a=(e^z-e^(-z))/(e^z+e^(-z)

        proof using derivative of u/v:
        [(e^z+e^(-z))*d(e^z-e^(-z))]-[(e^z-e^(-z))*d((e^z+e^(-z))]/[(e^z+e^(-z)]**2
        =[(e^z+e^(-z))*(e^z+e^(-z))]-[(e^z-e^(-z))*(e^z-e^(-z))]/[(e^z+e^(-z)]**2
        =[(e^z+e^(-z)]**2-[(e^z-e^(-z)]**2/[(e^z+e^(-z)]**2
        =1-[(e^z-e^(-z))/(e^z+e^(-z)]**2
        =1-a**2
        '''
        return delta * (1 - input**2)
    
    def reset_parameters(self):
        pass
    
class Sigmoid(Module):
    '''
    Also known as the logistic activation function
    Sig(X) = 1 / (1 + e^-x)
    0 <= Sig(X) <= 1
    '''
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        ''' no gradient to reset to zero '''
        pass

    def forward(self, X):
        # print((1 / (1 + np.exp(-X))).shape)
        self.last_input = X # keep track of the last input of the module
        return 1 / (1 + np.exp(-X))
    
    def update_parameters(self, learning_rate):
        ''' no parameters to update '''
        pass

    def backward_update_gradient(self, input, delta):
        ''' no gradient to update '''
        pass

    def backward_delta(self, input, delta):
        '''
        proof using derivative of u/v:
        ∂Sig = [(1+exp(-x)(d(1))-d(1+exp(-x)*1]/(1+exp(-x))**2
        since ∂(1+exp(-x))=∂(1)+∂(exp(-x))=-exp(-x),
        ∂Sig = exp(-x)/(1+exp(-x))**2
        ∂Sig = [1/(1+exp(-x))]*[1-(1/(1+exp(-x))]
        ∂Sig = Sig(x) * (1 - Sig(x))
        '''
        sig = 1/(1+np.exp(-input))
        return delta * (sig * (1 - sig))
    
    def reset_parameters(self):
        pass