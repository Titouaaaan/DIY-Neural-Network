from module import Loss
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

        return np.linalg.norm(y-yhat) ** 2

    def backward(self, y, yhat):
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
        return (-2 * (y - yhat)) 