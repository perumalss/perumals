import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
data = np.random.rand(100, 2)  # 100 points in 2D space

# ---- K-MEANS CLUSTERING ----
print("K-Means Clustering")

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, init='random', max_iter=100, n_init=1, random_state=42)

# Fit the model to data
kmeans.fit(data)

# Initial clusters (only displaying the first few points)
initial_centroids = kmeans.cluster_centers_
print("Initial Centroids:\n", initial_centroids)

# Final clusters
final_clusters = kmeans.labels_
print("Final Clusters for K-Means:\n", final_clusters)

# Number of epochs/iterations used
print("Epochs used by K-Means:", kmeans.n_iter_)

# Error Rate (Inertia - sum of squared distances to nearest cluster center)
kmeans_error = kmeans.inertia_
print("Error Rate (Inertia) for K-Means:", kmeans_error)

# Plot K-Means clustering
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=final_clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# ---- AGGLOMERATIVE CLUSTERING ----
print("\nAgglomerative Clustering")

# Initialize Agglomerative Clustering with 3 clusters
agglo = AgglomerativeClustering(n_clusters=3)

# Fit the model to data and predict labels
agglo_labels = agglo.fit_predict(data)

# Final clusters for Agglomerative Clustering
print("Final Clusters for Agglomerative Clustering:\n", agglo_labels)

# Calculate error rate as the mean squared error from data points to cluster centers
# For agglomerative, we'll assign cluster centers based on the mean of points in each cluster
centroids = np.array([data[agglo_labels == i].mean(axis=0) for i in range(3)])
agglo_error = np.sum([mean_squared_error(data[agglo_labels == i], [centroids[i]]*len(data[agglo_labels == i])) for i in range(3)])
print("Error Rate (MSE) for Agglomerative Clustering:", agglo_error)

# Plot Agglomerative Clustering
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=agglo_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], color='red', marker='X', s=200)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
