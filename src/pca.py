import numpy as np

class PCA:
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