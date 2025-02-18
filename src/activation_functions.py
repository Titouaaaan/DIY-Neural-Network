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
        return (1 - np.square(np.tanh(input))) * delta
    
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

class Softmax(Module):
    '''
    Computes the exponential of every score and 
    normalizes them (divide by the sum of the exponentials)
    Predicts the class with the highest probability
    '''
    def __init__(self):
        super().__init__()
    
    def zero_grad(self):
        ''' no gradient to reset to zero '''
        pass

    def forward(self, X):
        '''
        Softmax Yhatk = exp(Xk) / j∑K exp(Xj)
        '''
        self.last_input = X
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def update_parameters(self, learning_rate):
        ''' no parameters to update '''
        pass

    def backward_update_gradient(self, input, delta):
        ''' no gradient to update '''
        pass

    def backward_delta(self, input, delta):
        '''
        Derivative of softmax:
        ∂S = ∂Si / ∂aj
        = (exp(ai) * ∑ - exp(aj)exp(ai)) / ∑exp(ak)**2
        = (exp(ai)/∑exp(ak)**2) * (∑ - exp(aj) / ∑exp(ak)**2)
        = Si * ( 1 - Sj )
        So if the loss function is cross entropy we don't actually need this!
        Why?
        Because the backprop step of the cross entropy actually takes into account the
        softmax gradient 
        So we don't need this unless the loss isn't cross entropy
        '''
        # return delta * (input * (1 - input))
        return delta
    
    def reset_parameters(self):
        pass