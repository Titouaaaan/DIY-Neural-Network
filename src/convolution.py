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
        - We first extract the dimensions of the input.
        - We calculate the output length based on the kernel size and stride.
        - We then create a matrix `input_col` that represents the flattened input patches.
        - We perform the convolution operation by multiplying `input_col` with the reshaped weights.
        - We add the bias term and reshape the output to the desired dimensions.
        '''
        batch_size, length, chan_in = input.shape
        assert chan_in == self.chan_in, "Input channel mismatch."

        # Store the input for use in the backward pass
        self.last_input = input

        # Calculate the output length
        self.out_length = (length - self.k_size) // self.stride + 1

        # Create the input_col matrix by extracting patches from the input
        self.input_col = (input[:,
            (self.stride * np.arange(self.out_length))[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :],
            :].reshape((batch_size, self.out_length, -1)))

        # Perform the convolution operation
        Z = self.input_col @ self.weights.reshape(self.chan_out, -1).T

        # Add the bias term
        Z += self.bias.T

        # Reshape the output to the desired dimensions
        Z = Z.reshape(batch_size, self.out_length, self.chan_out)
        return Z

    def backward_update_gradient(self, input, delta):
        '''
        - We first extract the batch size from the delta.
        - We update the bias gradient by taking the mean of delta across the batch and spatial dimensions.
        - We update the weight gradient by multiplying the reshaped delta with the reshaped input_col matrix.
        - We reshape the result to match the dimensions of the weights and divide by the batch size.
        '''
        batch_size = delta.shape[0]

        # Update the bias gradient
        self.gradient_bias += delta.mean(axis=(0, 1))

        # Update the weight gradient
        self.gradient_weights += (
            delta.transpose(2, 0, 1).reshape(self.chan_out, -1) @  # Reshape delta for matrix multiplication
            self.input_col.reshape(-1, self.k_size * self.chan_in)  # Reshape input_col for matrix multiplication
        ).reshape(self.weights.shape) / batch_size  # Reshape to match weights and divide by batch size

    def backward_delta(self, input, delta):
        '''
        - We first extract the batch size from the delta.
        - We perform a matrix multiplication between delta and the reshaped weights.
        - We reshape the result to match the dimensions of the input patches.
        - We create a zero matrix delta X with the same shape as the input.
        - We accumulate the gradients in delta X using advanced indexing.
        '''
        batch_size = delta.shape[0]

        # Perform matrix multiplication and reshape the result
        tmp = ( (delta @ self.weights.reshape(self.chan_out, -1)) / batch_size
            ).reshape(batch_size, self.out_length, self.k_size, self.chan_in)

        # Create a zero matrix dX with the same shape as the input
        deltaX = np.zeros_like(input, dtype=float)

        # Accumulate the gradients in deltaX using advanced indexing
        deltaX[:, self.stride * np.arange(self.out_length)[:, np.newaxis] + np.arange(self.k_size)[np.newaxis, :]] += tmp[:, np.arange(self.out_length)]

        return deltaX

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        batch_size, length, chan_in = X.shape
        out_length = (length - self.k_size) // self.stride + 1

        X_view = np.lib.stride_tricks.sliding_window_view(X, (1, self.k_size, 1))[::1, :: self.stride, ::1]
        X_view = X_view.reshape(batch_size, out_length, chan_in, self.k_size)

        self.output = np.max(X_view, axis=-1)
        return self.output

    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        out_length = (length - self.k_size) // self.stride + 1

        input_view = np.lib.stride_tricks.sliding_window_view(input, (1, self.k_size, 1))[
            ::1, :: self.stride, ::1
        ]
        input_view = input_view.reshape(batch_size, out_length, chan_in, self.k_size)

        max_indices = np.argmax(input_view, axis=-1)

        # Create indices for batch and channel dimensions
        batch_indices, out_indices, chan_indices = np.meshgrid(
            np.arange(batch_size),
            np.arange(out_length),
            np.arange(chan_in),
            indexing="ij",
        )

        # Update d_out using advanced indexing
        self.d_out = np.zeros_like(input)
        self.d_out[
            batch_indices, out_indices * self.stride + max_indices, chan_indices
        ] += delta[batch_indices, max_indices, chan_indices]

        return self.d_out

class Flatten(Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)