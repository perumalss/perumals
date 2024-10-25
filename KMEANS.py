import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data for clustering
data = np.random.rand(100, 2)  # 100 points in 2D space

# K-Means Clustering
kmeans = KMeans(n_clusters=3, init='random', max_iter=100, n_init=1, random_state=42)
kmeans.fit(data)

# Initial Clusters (First 5 points to get a sample)
print("Initial clusters for K-Means:", kmeans.labels_[:5])

# Final Clusters with Epochs
print("Final clusters for K-Means:", kmeans.labels_)
print("Epochs:", kmeans.n_iter_)

# Error Rate for K-Means
kmeans_error = kmeans.inertia_
print("Error Rate (Inertia) for K-Means:", kmeans_error)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(data)

# Final Clusters for Agglomerative
print("Final clusters for Agglomerative:", agglo_labels)
