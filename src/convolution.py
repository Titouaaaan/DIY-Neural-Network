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
    def __init__(self, k_size, chan_in, chan_out, stride):
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self.weights = np.random.rand(k_size, chan_in, chan_out) - 0.5
        self.bias = np.random.rand(chan_out) - 0.5
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
    
    def forward(self, input):
        '''
        create `input_col`, the flattened input patches.
        perform the convolution operation by multiplying `input_col` with the reshaped weights.
        add the bias term and reshape the output
        '''
        batch_size, length, _ = input.shape

        # Store the input for use in the backward pass
        self.last_input = input

        # Calculate the output length
        self.out_length = (length - self.k_size) // self.stride + 1

        # Create the input_col matrix by extracting patches from the input
        self.input_col = (input[:,
            (self.stride * np.arange(self.out_length))[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :],
            :].reshape((batch_size, self.out_length, -1)))

        # Perform the convolution operation
        Z = np.dot(self.input_col,self.weights.reshape(self.chan_out, -1).T) + self.bias.T

        # Reshape the output to the desired dimensions
        return Z.reshape(batch_size, self.out_length, self.chan_out)
        

    def backward_update_gradient(self, input, delta):
        '''
        update the bias gradient by taking the mean of delta across the batch and spatial dimensions.
        update the weight gradient by multiplying the reshaped delta with the reshaped input_col matrix.
        reshape the result to match the dimensions of the weights and divide by the batch size.
        '''
        batch_size = delta.shape[0]

        # Update the bias gradient
        self.gradient_bias += delta.mean(axis=(0, 1))

        # Update the weight gradient
        self.gradient_weights += (
            np.dot(
                delta.transpose(2, 0, 1).reshape(self.chan_out, -1),  # Reshape delta for matrix multiplication
                self.input_col.reshape(-1, self.k_size * self.chan_in)  # Reshape input_col for matrix multiplication
                )
                ).reshape(self.weights.shape) / batch_size  # Reshape to match weights and divide by batch size

    def backward_delta(self, input, delta):
        '''
        perform a matrix multiplication between delta and the reshaped weights.
        reshape the result to match the dimensions of the input patches.
        accumulate the gradients
        '''
        batch_size = delta.shape[0]

        # Perform matrix multiplication and reshape the result
        grad = ( (np.dot(delta,self.weights.reshape(self.chan_out, -1))) / batch_size
            ).reshape(batch_size, self.out_length, self.k_size, self.chan_in)

        # Create a zero matrix dX with the same shape as the input
        delta = np.zeros_like(input, dtype=float)

        # Accumulate the gradients in delta using advanced indexing
        delta[:, self.stride * np.arange(self.out_length)[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :]] += grad[:, np.arange(self.out_length)]

        return delta

    def update_parameters(self, learning_rate):
        max_gradient = 1.0
        np.clip(self.gradient_weights, -max_gradient, max_gradient, out=self.gradient_weights)
        np.clip(self.gradient_bias, -max_gradient, max_gradient, out=self.gradient_bias)

        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias

class Flatten(Module):
    def forward(self, input):
        '''
        Reshape the input to flatten all dimensions except the batch size.
        '''
        self.last_input = input
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)
    
    def backward_delta(self, input, delta):
        '''
        just reshape the gradient to match the original input shape
        '''
        return delta.reshape(input.shape) * input

    def zero_grad(self):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def update_parameters(self, learning_rate):
        pass