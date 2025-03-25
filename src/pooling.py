from src.module import Module
import numpy as np

class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        '''
        - k_size: Size of the pooling window.
        - stride: Stride of the pooling operation.
        '''
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, input):
        '''
        - Compute the output length based on the input length, kernel size, and stride.
        - Reshape the input to extract patches for pooling.
        - Compute the argmax indices for the backward pass.
        '''
        self.last_input = input
        batch_size, length, chan_in = input.shape
        self.out_length = (length - self.k_size) // self.stride + 1
        
        # Reshape the input to extract patches for pooling
        reshaped_input = (input[:, 
            (self.stride * np.arange(self.out_length))[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :], 
            :].reshape((batch_size, self.out_length, self.k_size, chan_in)))
        # Compute the argmax indices for the backward pass
        self.amax = 1. * (reshaped_input == np.amax(reshaped_input, axis=2, keepdims=True))
        Z = reshaped_input.max(axis=2) # max pool
        return Z
 
    def backward_delta(self, input, delta):
        '''
        gradient of the loss with respect to the input
        use the argmax indices to propagate the gradient back to the input.
        '''
        batch_size, _, chan_in = input.shape
        grad = (
            np.tile(delta, 2) * 
            self.amax.reshape(batch_size, self.out_length, -1)
        ).reshape(batch_size, self.out_length, self.k_size, chan_in) / batch_size
        deltax = np.zeros_like(input,dtype = np.float64)
        deltax[:, self.stride * np.arange(self.out_length)[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :]] += grad[:, np.arange(self.out_length)]
    
        return deltax

    def zero_grad(self):
        pass  

    def backward_update_gradient(self, input, delta):
        pass  

    def update_parameters(self, learning_rate):
        pass  

r''' 
I think this doesnt work, so I need to fix it so don't use it atm
As long as this comment is here then its broken ¯\(ツ)/¯
'''
class AveragePool1D(Module):
    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, input):
        self.last_input = input
        batch_size, length, chan_in = input.shape
        self.out_length = (length - self.k_size) // self.stride + 1

        reshaped_input = (input[:,
            (self.stride * np.arange(self.out_length))[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :],
            :].reshape((batch_size, self.out_length, self.k_size, chan_in)))

        Z = reshaped_input.mean(axis=2)
        return Z

    def backward_delta(self, input, delta):
        batch_size, _, chan_in = input.shape
        grad = (delta / self.k_size).reshape(batch_size, self.out_length, 1, chan_in)
        deltax = np.zeros_like(input, dtype=np.float64)
        deltax[:, self.stride * np.arange(self.out_length)[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :]] += grad

        return deltax

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, learning_rate):
        pass

class MaxPool2D(Module):
    '''
    SAME AS CONV2D SO FAR
    THIS IS NOT THE FINAL PRODUCT. I USED A BUNCH OF FOR LOOPS WHICH MAKES THE TRAINING
    SUPER SLOW (unless you have a crazy good cpu)
    IF YOU SEE THIS COMMENT THEN ITS STILL NOT OPTIMIZED SO DONT TRAIN YOUR CNN
    WITH THOUSANDS OF IMAGES IF YOU DONT WANT IT TO TAKE HOURS
    I WILL FIX IT I SWEAR
    '''
    def __init__(self, k_size, stride):
        '''
        - k_size: Size of the pooling window.
        - stride: Stride of the pooling operation.
        '''
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, input):
        '''
        Perform the forward pass of the max pooling operation.

        - input: Input tensor of shape (batch_size, height, width, channels).
        '''
        batch_size, height, width, channels = input.shape

        # Store the input for use in the backward pass
        self.last_input = input

        # Calculate the output dimensions
        self.out_height = (height - self.k_size) // self.stride + 1
        self.out_width = (width - self.k_size) // self.stride + 1

        # Initialize the output tensor
        output = np.zeros((batch_size, self.out_height, self.out_width, channels))

        # Perform the max pooling operation
        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.out_height):
                    for j in range(self.out_width):
                        # Extract the patch
                        patch = input[b, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, c]
                        # Find the max value in the patch
                        output[b, i, j, c] = np.max(patch)

        return output

    def backward_delta(self, input, delta):
        '''
        Gradient of the loss with respect to the input.
        Use the argmax indices to propagate the gradient back to the input.
        '''
        batch_size, _, _, channels = input.shape

        # Create a zero matrix for the gradient
        deltax = np.zeros_like(input, dtype=np.float64)

        # Perform the backward pass of the max pooling operation
        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.out_height):
                    for j in range(self.out_width):
                        # Extract the patch
                        patch = input[b, i*self.stride:i*self.stride+self.k_size, j*self.stride:j*self.stride+self.k_size, c]
                        # Find the index of the max value in the patch
                        max_index = np.unravel_index(np.argmax(patch), patch.shape)
                        # Place the gradient in the correct position
                        deltax[b, i*self.stride+max_index[0], j*self.stride+max_index[1], c] = delta[b, i, j, c]

        return deltax

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, learning_rate):
        pass
