from src.encapsulation import Sequential
from src.loss import Loss
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder():
    '''
    The idea of the auto encoder is to have an encoder and a decoder
    the encoder reduces the dimensions of the original data, into what we call the latent space 
    the decoder reconstructs an image from a compressed, lower dimension data (image)
    The pipeline looks like this:
    X -> Encoder -> Z -> Decoder -> Xhat
    with X being the input data, Z the latent space, Xhat the 'prediction', reconstructed image
    '''
    def __init__(self, encoder: Sequential, decoder: Sequential, loss: Loss, learning_rate: float):
        self.encoder = encoder # encoder network
        self.decoder = decoder # decoder network
        self.loss_function = loss # loss function
        self.learning_r = learning_rate # learning rate for param update
    
    def forward(self, X):
        encoded = self.encoder.forward(X)
        decoded = self.decoder.forward(encoded)
        return decoded
    
    def backward(self, X, decoded):
        loss = self.loss_function.forward(X, decoded) # calculate the loss between the original image and the reconstructed one
        gradient = self.loss_function.backward(X, decoded) # calculate the gradient of the loss from the reconstructed image

        gradient = self.decoder.backward(gradient)  # backprop the gradient of the loss into the decoder
        self.encoder.backward(gradient) # backprop the gradient of the decoder into the encoder (reverse path of the pipeline we talked about above)

        return loss # return the loss for logging

    def update_parameters(self):
        self.encoder.update_parameters(self.learning_r)
        self.decoder.update_parameters(self.learning_r)

    def train_auto_encoder(self, data: tuple, batch_size: int, max_epoch: int, show_results:bool=True):
        X_train, X_test = data
        n_samples = X_train.shape[0] 
        log_loss = []
        
        for epoch in range(max_epoch):
            indices = np.random.permutation(n_samples) # generate a random array of indices

            # shuffle the train dataset
            X_epoch = X_train[indices] 

            # create the mini batches for this epoch
            X_batches = np.array_split(X_epoch, np.ceil(n_samples / batch_size)) # np.ceil rounds up the division, np.ceil(3.125) → 4 gives 4 batches

            epoch_loss = [] # log the losses of the epoch
            for minibatch in X_batches: # iterate through the mini batches
                decoded = self.forward(minibatch)

                loss = self.backward(minibatch, decoded)

                self.update_parameters()

                epoch_loss.append(loss)

            log_loss.append(np.mean(epoch_loss))
        
            print(f'epoch {epoch}, loss:{np.mean(epoch_loss)}')
        
        if show_results:
            decoded_images = self.forward(X_test[:20])  # Get the first 20 test images

            fig, axes = plt.subplots(2, 20, figsize=(20, 4))  # 2 rows, 20 columns
            
            for i in range(20):
                # Original image (top row)
                axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
                axes[0, i].axis('off')

                # Reconstructed image (bottom row)
                axes[1, i].imshow(decoded_images[i].reshape(28, 28), cmap='gray')
                axes[1, i].axis('off')

            plt.suptitle("Top: Original Images | Bottom: Reconstructed Images")
            plt.show()

        return log_loss