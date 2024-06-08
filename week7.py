import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# KMeans Algorithm
def kmeans(X, n_clusters, max_iters=100):
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Gaussian Mixture Model Algorithm
def gmm(X, n_components, max_iters=100):
    # Initialize parameters
    n_samples, n_features = X.shape
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.array([np.cov(X.T) for _ in range(n_components)])
    responsibilities = np.zeros((n_samples, n_components))

    # EM algorithm
    for _ in range(max_iters):
        # E-step: Compute responsibilities
        for i in range(n_samples):
            for k in range(n_components):
                responsibilities[i, k] = weights[k] * multivariate_normal(X[i], means[k], covariances[k])
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]

        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n_samples
        means = (responsibilities.T @ X) / Nk[:, np.newaxis]
        covariances = np.array([((responsibilities[:, k] * (X - means[k]).T) @ (X - means[k])) / Nk[k] for k in range(n_components)])

    return responsibilities.argmax(axis=1), means

# Multivariate normal PDF
def multivariate_normal(x, mean, cov):
    return np.exp(-0.5 * (x - mean) @ np.linalg.inv(cov) @ (x - mean).T) / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** len(x))

# Load Iris dataset
def load_iris():
    data = np.genfromtxt("iris.csv", delimiter=",", skip_header=1)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    return X, y

def main():
    st.title("Iris Clustering Visualization")

    # Load the Iris dataset
    X, y = load_iris()

    # Fit KMeans
    kmeans_labels, kmeans_centroids = kmeans(X, n_clusters=3)

    # Fit Gaussian Mixture Model
    gmm_labels, gmm_means = gmm(X, n_components=3)

    # Plotting the results
    plt.figure(figsize=(14, 7))

    # Plot KMeans clusters
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', marker='x', s=200)
    plt.title('KMeans Clustering')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')

    # Plot Gaussian Mixture Model clusters
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(gmm_means[:, 0], gmm_means[:, 1], c='red', marker='x', s=200)
    plt.title('Gaussian Mixture Model Clustering')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')

    # Display the plots
    st.pyplot()

if __name__ == "__main__":
    main()
