import numpy as np
from src.activation_functions import Softmax

class DimensionReduction():
    def __init__(self):
        pass

    def fit_transform(self, X):
        pass

class PCA(DimensionReduction):
    def __init__(self, n_components):
        """
        Initialize the PCA with the number of principal components to keep.

        Parameters:
        - n_components: Number of principal components to keep.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Parameters:
        - X: Data matrix of shape (n_samples, n_features).
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transform the data to the principal component space.

        Parameters:
        - X: Data matrix of shape (n_samples, n_features).

        Returns:
        - X_pca: Transformed data matrix of shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        X_pca = np.dot(X_centered, self.components)
        return X_pca

    def fit_transform(self, X):
        """
        Fit the PCA model to the data and transform it to the principal component space.

        Parameters:
        - X: Data matrix of shape (n_samples, n_features).

        Returns:
        - X_pca: Transformed data matrix of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)
    
class TSNE(DimensionReduction):
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        """
        Initialize the t-SNE algorithm.

        Parameters:
        - n_components: Number of dimensions to reduce to (default is 2 for visualization).
        - perplexity: The perplexity parameter, which can be thought of as a smooth measure of the effective number of neighbors.
        - learning_rate: The learning rate for gradient descent.
        - n_iter: Number of iterations to run the optimization.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def _compute_pairwise_affinities(self, X):
        """
        Compute the pairwise affinities in the high-dimensional space using a Gaussian kernel.

        Formula:
        p_{j|i} = exp(-||x_i - x_j||^2 / (2 * sigma_i^2)) / ∑_{k!=i} exp(-||x_i - x_k||^2 / (2 * sigma_i^2))
        """
        n = X.shape[0]
        sum_X = np.sum(np.square(X), axis=1)
        distances = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        sigma_sq = np.mean(distances)  # Simplified sigma calculation
        affinities = np.exp(-distances / (2 * sigma_sq))
        np.fill_diagonal(affinities, 0)
        softmax = Softmax()
        affinities = softmax.forward(affinities)
        affinities = (affinities + affinities.T) / (2 * n)
        return affinities

    def _compute_low_dimensional_affinities(self, Y):
        """
        Compute the pairwise affinities in the low-dimensional space using a Student's t-distribution.

        Formula:
        q_{ij} = (1 + ||y_i - y_j||^2)^-1 / ∑_{k!=l} (1 + ||y_k - y_l||^2)^-1
        """
        n = Y.shape[0]
        sum_Y = np.sum(np.square(Y), axis=1)
        distances = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
        affinities = 1 / (1 + distances)
        np.fill_diagonal(affinities, 0)
        affinities = affinities / np.sum(affinities)
        return affinities

    def fit_transform(self, X):
        """
        Perform t-SNE dimensionality reduction.

        Parameters:
        - X: High-dimensional data (n_samples, n_features).

        Returns:
        - Y: Low-dimensional embedding (n_samples, n_components).
        """
        n = X.shape[0]
        Y = np.random.randn(n, self.n_components)

        # Compute high-dimensional affinities
        P = self._compute_pairwise_affinities(X)

        for iteration in range(self.n_iter):
            print(f'iteration: {iteration}/{self.n_iter}')
            # Compute low-dimensional affinities
            Q = self._compute_low_dimensional_affinities(Y)

            # Compute gradient using matrix operations
            PQ = P - Q
            grad = np.dot((PQ - PQ.T), Y)

            # Update the low-dimensional embedding
            Y = Y + self.learning_rate * grad

            # Optionally, add momentum and gain adjustments for better convergence

        return Y