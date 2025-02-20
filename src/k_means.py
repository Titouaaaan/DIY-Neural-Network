import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KMeans:
    def __init__(self, k, points, max_epoch=100, eps=1e-4):
        '''
        k: amount of clusters (classes)
        points: data points to cluster
        max_epoch: how many training epochs
        eps: used to check distance between centroids for convergence
        '''
        self.k = k
        self.points = points
        self.max_epoch = max_epoch
        self.eps = eps

    def cluster(self):
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
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.points[:, 0], self.points[:, 1], c=assigned_labels, cmap='viridis', alpha=0.6)

        # Add color bar to show label mapping
        legend = plt.colorbar(scatter, ticks=np.unique(assigned_labels))
        legend.set_label('Assigned Labels')

        # Plot cluster centroids with class numbers
        for i in range(self.k):
            plt.text(centroids[i, 0], centroids[i, 1], str(i), fontsize=12, ha='center', va='center', color='red')

        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
        plt.title('K-means Clustering with Assigned Labels')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def run_clustering(self, y_test):
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