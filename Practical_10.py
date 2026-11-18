# Practical No. 10
# Title: Apply K-means & visualize clusters, use silhouette score & elbow method to tune k

import numpy as np
import matplotlib.pyplot as plt

print("Libraries Imported Successfully")

np.random.seed(42)

X = np.vstack([
    np.random.randn(50, 2) + [5, 3],
    np.random.randn(50, 2) + [6, 4],
    np.random.randn(50, 2) + [7, 2]
])

print("Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)

def kmeans(X, k):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for i in range(100):
        distances = np.linalg.norm(
            X[:, np.newaxis] - centroids,
            axis=2
        )
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == j].mean(axis=0)
            for j in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

for k in range(2, 7):
    labels, centers = kmeans(X, k)
    inertia = sum(
        np.sum((X[labels == j] - centers[j]) ** 2)
        for j in range(k)
    )
    print("K =", k, "Inertia =", round(inertia, 2))

labels, centers = kmeans(X, 3)
print("K-Means Model Trained Successfully")

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap='rainbow'
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color='black',
    marker='X',
    s=150
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
Libraries Imported Successfully
Dataset Loaded Successfully
Dataset Shape: (150, 2)
K = 2 Inertia = 310.38
K = 3 Inertia = 203.83
K = 4 Inertia = 170.12
K = 5 Inertia = 127.72
K = 6 Inertia = 112.23
K-Means Model Trained Successfully
"""
