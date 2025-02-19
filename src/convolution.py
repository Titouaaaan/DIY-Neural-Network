import numpy as np
from src.module import Module

class Convolution1D(Module):
    '''
    Convolution -> a convolutional neural network (CNN) is a kind of neural network
    that tries to learn a representation of images 
    We want to apply convolution layers on an image to get 'feature maps' (more below),
    followed by pooling layers, to undersample the output of the previous layers and reduce the dimensions,
    and finally fully connected layers, applied on the flatted learned representations of 
    the conv and pooling layers

    In our 1D case:
    Input image 2D size NxM, C (color channels) => flattened image 1D size (size=NxM), C => convolution layers => pooling layers => FC layers

    this class is a module so we can use it in a sequential architecture (it will be the same for the pooling layers)
    '''
    def __init__(self, ksize:int, stride: int, c_out:int, channels:int):
        self.ksize = ksize # kernel size of filter 
        self.stride = stride # step, ie. how many pixels do we shift after an operation (ex: with s=2 ■■■□□□□ -> □□■■■□□ -> □□□□■■■)
        self.c_out = c_out # amount of filters we want
        self.channels = channels # amount of channels, 1 for grayscale, 3 for RGB etc...
        self.filters = np.random.randn(self.c_out, self.ksize, self.channels) * 0.01 # small constant to have small weights (avoid working with large values)
        self.bias = np.random.randn(self.c_out) * 0.01 # small bias
        self.filters_gradient = np.zeros_like(self.filters) # init the gradients as zeros for the moment
        self.bias_gradient = np.zeros_like(self.bias) # same 
    
    def zero_grad(self):
        '''
        reset the gradients
        '''
        self.filters_gradient = np.zeros_like(self.filters)
        self.bias_gradient = np.zeros_like(self.bias)

    def forward(self, X):
        '''
        X represents the flattened images
        '''
        return super().forward(X)
    
    def backward_delta(self, input, delta):
        return super().backward_delta(input, delta)
    
    def backward_update_gradient(self, input, delta):
        return super().backward_update_gradient(input, delta)
    
    def update_parameters(self, learning_rate=0.001):
        '''
        Update the filters and the biases 
        '''
        self.filters -= learning_rate * self.filters_gradient
        self.bias -= learning_rate * self.bias_gradient
    