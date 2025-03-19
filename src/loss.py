from src.module import Loss
import numpy as np

class MSELoss(Loss):
    '''
    MSE(y, yhat) = 1/n * ∑(y-yhat) ** 2
    The term (y-yhat) ** 2 calculates the squared difference 
    between the actual value and the predicted value for each example.
    Squaring the difference ensures that the loss is always non-negative and
    gives more weight to larger errors, making the model more sensitive to outliers.
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

        The term (yi * log(pi) is active only for the true class (where yi = 1).
        It measures how well the model predicts the probability of the true class.
        The log function ensures that the loss is large when the predicted probability pi is small (i.e., the model is confidently wrong).
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
        epsilon = 1e-12
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return yhat - y
    
class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        '''
        True Positive (y = 1): When the true label is 1, the loss simplifies to -log(p). 
        This means the loss decreases as the predicted probability p approaches 1.

        True Negative (y = 0): When the true label is 0, the loss simplifies to -log(1-p). 
        This means the loss decreases as the predicted probability p approaches 0.

        BCE(y, yhat) = -(y * log(yhat) + (1 - y)log(1 - yhat))
        '''
        epsilon = 1e-12 # small constant to avoid having log(0) which is -inf which leads to nan
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return - np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) # we do np.mean to get the avg of losses of the batch
    
    def backward(self, y, yhat):
        '''
        ∂L/ ∂yhat , term by term:
        ∂yhat of (- y * log(yhat)) = -y*1/yhat = -y/yhat 
        ∂yhat of (-(1-y)*log(1-yhat)) = -(1-y) * (-1/1-yhat) = (1-y)/(1-yhat)
        so 
        ∂L/∂yhat = -y/yhat + (1-y)/(1-yhat)
                 = -(y(1-yhat)/yhat(1-yhat)) + yhat(1-y)/yhat(1-yhat)
                 = (-y+yyhat+yhat-yyhat) / yhat(1-yhat) 
                 = (yhat - y) / yhat * (1-yhat)
        and since the BCE is averaged over the batch, we technically have 
        1/N * ∑ (yhat - y) / yhat * (1-yhat) 
        so we divide by N, i.e multiply the denominator by N=yhat.shape[0] the batch size
        '''
        epsilon = 1e-12 
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return (yhat - y) / (yhat * (1 - yhat) * yhat.shape[0])

class CategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        '''
        Categorical Cross-Entropy (CCE) is used for multi-class classification problems.
        It measures the difference between the true label y and the prediction yhat (both distributions)

        For each class, the loss is calculated as -log(p_true_class), where p_true_class is the predicted probability of the true class
        This means the loss decreases as the predicted probability of the true class approaches 1

        CCE(y, yhat) = -∑(y * log(yhat)) / N
        where N is the number of samples in the batch.
        '''
        epsilon = 1e-12  # small constant to avoid having log(0) which is -inf which leads to nan
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return -np.sum(y * np.log(yhat)) / yhat.shape[0]  # we do np.sum to get the total loss of the batch and divide by N to get the average

    def backward(self, y, yhat):
        '''
        ∂L/∂yhat, term by term:
        ∂yhat of (- y * log(yhat)) = -y * 1/yhat = -y/yhat
        Since Categorical Cross-Entropy is the sum over all classes, the gradient is simply the difference between the predicted probabilities and the true labels
        So,
        ∂L/∂yhat = yhat - y
        '''
        epsilon = 1e-12
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return (yhat - y)  # the gradient is simply the difference between the predicted probabilities and the true labels
