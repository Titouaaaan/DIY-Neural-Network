from src.module import Loss
import numpy as np

class MSELoss(Loss):
    '''
    MSE(y, yhat) = 1/n * ∑(y-yhat) ** 2
    '''
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        '''
        Calculation of MSE || y - yhat || ** 2
        Computes Mean Squared error/loss between targets and predictions. 
        Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
            targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
        Returns: scalar
        '''
        assert y.shape == yhat.shape, f'Mismatch of shapes: y->{y.shape} and yhat->{yhat.shape}'

        return np.mean((y-yhat) ** 2)

    def backward(self, y, yhat) -> np.array:
        '''
        Computes mean squared error gradient between targets and predictions. 
        Input: predictions (N, k) ndarray (N: no. of samples, k: no. of output nodes)
            targets (N, k) ndarray        (N: no. of samples, k: no. of output nodes)
        Returns: (N,k) ndarray

        ∂MSE / ∂yhat = 
        ∂/∂yhat = ∑(yi - yhati)**2)  
        ∂/∂yhati = (yi - yhati)**2 = 2(yi - yhati) * -1 = -2(yi - yhati)
        '''
        assert y.shape == yhat.shape, f'Mismatch of shapes: y->{y.shape} and yhat->{yhat.shape}'
        return (-2 * (y - yhat)) / y.shape[0]
    
class CrossEntropy(Loss):
    '''
    Used to calculate how different a predicted probability distribution (key word here!)
    to the true distribution
    It quantifies the uncertainty between predicted and actual values:
    high probability to the correct class -> low cross entropy, else high cross entropy (and we want to minimize the loss)
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, y, yhat):
        '''
        L = - ∑ (yi * log(pi))
        where:
        ∑ sum of i over C, where C is the number of classes
        yi is 1 for correct class i, 0 for rest (one hot encoding)
        pi is the predicted probability for class i
        '''
        epsilon = 1e-12 # small constant to avoid having log(0) which is -inf which leads to nan
        #yhat = np.clip(yhat, 1e-12, 1-1e-12)
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return -np.sum(y * np.log(yhat)) / y.shape[0]

    def backward(self, y, yhat):
        '''
        ∂L / ∂yhat = -yi * 1/yhati
        = - yi/yhati, so for a batch
        = -1/N * ∑ yji / yhatji

        now we need the derivative of the loss with regards to the activation function:
        ∂L / ∂zi (with zi being the softmax function)
        ∂L / ∂zi = ∑ (∂L / ∂yhatj) * (∂pj / ∂zi)
        = pi - yi
        '''
        #yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return yhat - y