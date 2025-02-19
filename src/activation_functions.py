import numpy as np
from src.module import Module

class TanH(Module):
    '''
    z = input
    tanh(z) = (e^z - e^-z) / (e^z + e^-z)

    Tanh (Hyperbolic Tangent) introduces non-linearity into the model
    It maps input values to the range (-1, 1), which helps in centering the data and can make the optimization process more efficient

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

    Very useful in binary classification problems.
    It maps input values to the range (0, 1), making it useful for output layers where probabilities are required
    Occasional problem: vanishing gradients, where gradients become very small, slowing down learning (This leads to nan appearing in the matrices)
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
    
    Usually used in the output layer of neural networks for multi-class classification problems
    converts raw scores into probabilities that sum to one, allowing the model to make a probabilistic prediction across multiple classes

    Softmax Yhatk = exp(Xk) / j∑K exp(Xj)

    Its pretty good because it puts a focus on the largest values and suppresses smaller ones, 
    making the output more interpretable as probabilities.
    '''
    def __init__(self):
        super().__init__()
    
    def zero_grad(self):
        ''' no gradient to reset to zero '''
        pass

    def forward(self, X):
        '''
        Softmax Yhatk = exp(Xk) / j∑K exp(Xj)

        About the X - np.max(X, axis=1, keepdims=True) trick:
        this adjustment does not change the outcome of the Softmax function because it affects all elements in the same row equally, 
        which preserves the relative differences between them.

        You can do the math and verify that if we remove a constant C to:
        exp(Xk) / j∑K exp(Xj) which becomes exp(Xk - C) / j∑K exp(Xj - C)
        we actually get:
        exp(Xk - C) / j∑K exp(Xj - C) = (exp(Xk) exp(- C)) / j∑K (exp(Xj) exp(- C)) = exp(Xk) / j∑K exp(Xj) 
        '''
        self.last_input = X # as usual we keep track of the input
        exps = np.exp(X - np.max(X, axis=1, keepdims=True)) # exponential of each input element, minus the maximum value in each row (axis=1) for stability
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