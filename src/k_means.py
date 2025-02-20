import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KMeans:
    def __init__(self, k, points, max_epoch=100, eps=1e-4):
        '''
        Initialize the KMeans clustering algorithm.

        Parameters:
        - k: Number of clusters to form.
        - points: Data points to cluster, represented as a 2D numpy array where each row is a data point.
        - max_epoch: Maximum number of iterations to run the algorithm.
        - eps: Convergence threshold; the algorithm stops if the change in centroids is less than this value.
        '''
        self.k = k
        self.points = points
        self.max_epoch = max_epoch
        self.eps = eps

    def cluster(self):
        '''
        Perform the KMeans clustering algorithm.

        Steps:
        1. Initialize centroids randomly from the data points.
        2. Iteratively update cluster assignments and centroids until convergence.

        Mathematical Concepts:
        - Euclidean Distance: Used to measure the distance between data points and centroids.
        - Centroid: The mean of all points assigned to a cluster, representing the cluster's center.

        Returns:
        - labels: Cluster assignments for each data point.
        - centroids: Final centroids of the clusters.
        '''
        random_indices = np.random.choice(self.points.shape[0], self.k, replace=False) # random indices to choose the initial centroids
        centroids = self.points[random_indices]

        for _ in range(self.max_epoch):
            old_centroids =  centroids # store the previous centroids

            distances = np.linalg.norm(self.points[:, np.newaxis] - centroids, axis=2) # calculate the distance between each point and each centroid
            labels = np.argmin(distances, axis=1) # assign each point to the closest centroid
            
            centroids = np.array([self.points[labels == i].mean(axis=0) for i in range(self.k)]) # update the centroids based on their assigned clusters

            # Check for convergence
            if np.linalg.norm(centroids - old_centroids) < self.eps:
                break

        return labels, centroids
    
    def plot_clusters(self, labels, centroids, assigned_labels):
        '''
        Visualize the clusters formed by the KMeans algorithm.

        Parameters:
        - labels: Cluster assignments for each data point.
        - centroids: Centroids of the clusters.
        - assigned_labels: Labels assigned to each cluster based on majority class.

        Steps:
        1. Plot the data points colored by their assigned labels.
        2. Plot the centroids with class numbers.

        Visualization:
        - Scatter Plot: Data points are plotted in 2D space with colors representing their assigned labels.
        - Centroids: Marked with 'X' and labeled with the cluster index.
        '''
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.points[:, 0], self.points[:, 1], c=assigned_labels, cmap='viridis', alpha=0.6)

        # Add color bar to show label mapping
        legend = plt.colorbar(scatter, ticks=np.unique(assigned_labels))
        legend.set_label('Assigned Labels')

        # Plot cluster centroids with class numbers
        for i in range(self.k):
            plt.text(centroids[i, 0], centroids[i, 1], str(i), fontsize=12, ha='center', va='center', color='red')

        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red')
        plt.title('K-means Clustering with Assigned Labels')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def run_clustering(self, y_test):
        '''
        Run the KMeans clustering algorithm and visualize the results.

        Parameters:
        - y_test: True labels of the data points, used to assign majority class labels to clusters.

        Steps:
        1. Perform KMeans clustering.
        2. Assign labels to clusters based on the majority class in each cluster.
        3. Plot the clusters with assigned labels.

        Returns:
        - labels: Cluster assignments for each data point.
        '''
        labels, centroids = self.cluster()

        # Assign labels to clusters based on majority class in each cluster
        assigned_labels = np.zeros_like(labels)
        for cluster_id in range(self.k):
            mask = labels == cluster_id
            if np.any(mask):  # Check if the cluster is not empty
                majority_label = Counter(y_test[mask]).most_common(1)[0][0]
                assigned_labels[mask] = majority_label

        # Plot clusters with assigned labels
        self.plot_clusters(labels, centroids, assigned_labels)
        return labels