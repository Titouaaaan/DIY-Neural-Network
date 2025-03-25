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

class Convolution2D(Module):
    '''
    THIS IS NOT THE FINAL PRODUCT. I USED A BUNCH OF FOR LOOPS WHICH MAKES THE TRAINING
    SUPER SLOW (unless you have a crazy good cpu)
    IF YOU SEE THIS COMMENT THEN ITS STILL NOT OPTIMIZED SO DONT TRAIN YOUR CNN
    WITH THOUSANDS OF IMAGES IF YOU DONT WANT IT TO TAKE HOURS
    I WILL FIX IT I SWEAR
    '''
    def __init__(self, k_size, chan_in, chan_out, stride=1, padding=0):
        '''
        Initialize the Convolution2D layer.

        - k_size: Size of the convolutional kernel.
        - chan_in: Number of input channels.
        - chan_out: Number of output channels (filters).
        - stride: Stride of the convolution operation.
        - padding: Padding size around the input image.
        '''
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.weights = np.random.randn(k_size, k_size, chan_in, chan_out) * 0.1
        self.bias = np.random.randn(chan_out) * 0.1
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

    def forward(self, input):
        '''
        Perform the forward pass of the convolution operation.

        - input: Input tensor of shape (batch_size, height, width, channels).
        '''
        batch_size, height, width, _ = input.shape

        # Store the input for use in the backward pass
        # Reshape to ensure the channel dimension matches expected input
        self.last_input = input.reshape(batch_size, height, width, self.chan_in)

        # Calculate the output dimensions
        # The formula accounts for padding and stride to determine the size of the output feature map
        self.out_height = (height - self.k_size + 2 * self.padding) // self.stride + 1
        self.out_width = (width - self.k_size + 2 * self.padding) // self.stride + 1

        # Apply padding to the input if necessary
        if self.padding > 0:
            input_padded = np.pad(input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            input_padded = input

        # Initialize the output tensor
        output = np.zeros((batch_size, self.out_height, self.out_width, self.chan_out))

        # Perform the convolution operation
        # Iterate over each filter and each input channel
        for i in range(self.chan_out):
            for j in range(self.chan_in):
                for b in range(batch_size):
                    for x in range(0, height - self.k_size + 1, self.stride):
                        for y in range(0, width - self.k_size + 1, self.stride):
                            # Extract the patch from the input and perform element-wise multiplication with the filter
                            output[b, x // self.stride, y // self.stride, i] += np.sum(
                                input_padded[b, x:x + self.k_size, y:y + self.k_size, j] * self.weights[:, :, j, i]
                            )

        # Add the bias term
        output += self.bias

        return output

    def backward_update_gradient(self, input, delta):
        batch_size, out_height, out_width, chan_out = delta.shape

        # Correctly reshape delta
        delta_reshaped = delta.reshape(batch_size, out_height, out_width, chan_out)

        # Extract patches from input
        k_size = self.k_size
        chan_in = self.chan_in

        input_patches = np.zeros((batch_size, chan_in, k_size, k_size, out_height, out_width))

        for b in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    # Extract the patch and transpose to (chan_in, k_size, k_size)
                    # This aligns the input patches with the filter dimensions for gradient calculation
                    input_patches[b, :, :, :, i, j] = input[b, i:i+k_size, j:j+k_size, :].transpose(2, 0, 1)

        # Compute weight gradient using einsum for efficient tensor operations
        # The einsum operation sums over the batch, height, and width dimensions
        # bhwo and bcklhw are the axis labels for the two input tensors (delta_reshaped and input_patches)
        # sum over the axes b, h, w, and o (which are common between the two input tensors) 
        # and produce an output tensor with axes c, k, l, and o
        # for a clearer answer check this out: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        self.gradient_weights += np.einsum(
            'bhwo,bcklhw->cklo', 
            delta_reshaped, 
            input_patches
        ).transpose(1, 2, 0, 3) / batch_size  # Transpose to (k, l, c, o) <=> (k_size, k_size, chan_in, chan_out)

        # Compute bias gradient
        self.gradient_bias += np.sum(delta, axis=(0, 1, 2)) / batch_size



    def backward_delta(self, input, delta):
        '''
        Compute the gradient of the loss with respect to the input.
        '''
        batch_size, _, _, _ = delta.shape
        delta_input = np.zeros_like(input)

        # Rotate the weights for the convolution operation
        rotated_weights = np.rot90(self.weights, 2, axes=(0, 1))

        # Perform the convolution operation to compute the gradient
        for i in range(self.chan_in):
            for j in range(self.chan_out):
                for b in range(batch_size):
                    for x in range(delta_input.shape[1] - self.k_size + 1):
                        for y in range(delta_input.shape[2] - self.k_size + 1):
                            # Accumulate the gradient for the input by convolving with the rotated weights
                            delta_input[b, x:x + self.k_size, y:y + self.k_size, i] += delta[b, x, y, j] * rotated_weights[:, :, i, j]

        return delta_input

    def update_parameters(self, learning_rate):
        '''
        Update the parameters using gradient descent.
        '''
        self.weights -= learning_rate * self.gradient_weights
        self.bias -= learning_rate * self.gradient_bias

    def zero_grad(self):
        '''
        Reset the gradients to zero.
        '''
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

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