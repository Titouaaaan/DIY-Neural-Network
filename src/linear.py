from src.module import Module
import numpy as np

class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) 
        self.bias = np.random.randn(output_dim) 
        self.weights_gradient = np.zeros_like(self.weights) # same shape as weight but zeros
        self.bias_gradient = np.zeros_like(self.bias) # same but for bias
    
    def zero_grad(self) -> None:
        '''
        Reset the gradient of the weights and of the bias to zeros matrices
        '''
        self.weights_gradient = np.zeros_like(self.weights_gradient)
        self.bias_gradient = np.zeros_like(self.bias_gradient)

    def forward(self, X):
        '''
        Z = X @ W + b
        X: (batch, input_size)
        W: (input_size, output_size)
        b: (output_size, )
        '''
        assert X.shape[1] == self.weights.shape[0], f'Shape mismatch between X: {X.shape} and weights: {self.weights.shape}'
        assert self.bias.shape[0] == self.weights.shape[1], f'Shape mismatch between bias: {self.bias.shape} and weights: {self.weights.shape}'
        # print((X @ self.weights + self.bias).shape)
        return X @ self.weights + self.bias
    
    def update_parameters(self, learning_rate: float) -> None:
        '''
        Param update with gradient descent
        Substraction because gradients point in the direction of increasing loss (so we want to go in opposite direction)
        W = W - η * ∂L/∂W
        b = b - η * ∂L/∂b
        '''
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
    
    def backward_update_gradient(self, input, delta) -> None:
        '''
        Accumulate the gradients of weights (and bias) during back propagation
        input: (batch, input_dim) -> activation from previous layer
        delta δ: (batch, output_dim) -> gradient of loss with respect to this layer's output
        ∂L/∂W = W.T @ δ

        For bias,
        ∂L/∂b = ∑ * δi -> sum of i over batch size
        Summing over axis=0 (rows) collapses batch size, resulting in (output_dim,), which is the correct shape for bias_gradient
        Example, delta.sum(axis=0) on 
        delta = np.array([
            [0.2, -0.3],
            [-0.5,  0.7],
            [0.1,  0.4]
        ])
        returns [-0.2, 0.8]
        '''
        assert input.shape[1] == self.input_dim 
        assert delta.shape[1] == self.output_dim
        assert input.shape[0] == delta.shape[0], f'Batch shape [0] mismatch between input: {input.shape} and delta: {delta.shape}'
        self.weights_gradient = input.T @ delta
        self.bias_gradient = delta.sum(axis=0)

    def backward_delta(self, input, delta):
        '''
        Calculate the derivative of the error to propagate backward
        input: (batch, input_dim) -> activation from previous layer (n+1 layer forward speaking)
        delta δ: (batch, output_dim) -> gradient of loss with respect to this layer's output
        weights: (input_dim, output_dim)
        so delta @ weights.T:
        (batch ,output_dim) @ (output_dim, input_dim) = (batch, input_dim)
        '''
        assert delta.shape[1] == self.output_dim
        assert self.weights.shape[1] == delta.shape[1], f'Output dim mismatch between weights: {input.shape} and delta: {delta.shape}'
        return delta @ self.weights.T