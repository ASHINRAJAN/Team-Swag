import numpy as np
import streamlit as st

# K-means clustering algorithm
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    for _ in range(max_iters):
        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Gaussian Mixture Model (GMM) clustering algorithm
def gmm(X, k, max_iters=100):
    n_samples, n_features = X.shape
    # Initialize parameters
    weights = np.ones(k) / k
    means = X[np.random.choice(n_samples, k, replace=False)]
    covariances = np.array([np.cov(X.T) for _ in range(k)])
    responsibilities = np.zeros((n_samples, k))
    # EM algorithm
    for _ in range(max_iters):
        # E-step: Compute responsibilities
        for i in range(n_samples):
            for j in range(k):
                responsibilities[i, j] = weights[j] * multivariate_normal(X[i], means[j], covariances[j])
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n_samples
        means = (responsibilities.T @ X) / Nk[:, np.newaxis]
        covariances = np.array([((responsibilities[:, j] * (X - means[j]).T) @ (X - means[j])) / Nk[j] for j in range(k)])
    return responsibilities.argmax(axis=1), means

# Multivariate normal PDF
def multivariate_normal(x, mean, cov):
    return np.exp(-0.5 * (x - mean) @ np.linalg.inv(cov) @ (x - mean).T) / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** len(x))

def main():
    st.title("Iris Clustering Visualization")

    # Load the Iris dataset
    data = np.genfromtxt("iris.csv", delimiter=",", skip_header=1)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels

    # Perform K-means clustering
    kmeans_labels, kmeans_centroids = kmeans(X, k=3)

    # Perform Gaussian Mixture Model (GMM) clustering
    gmm_labels, gmm_means = gmm(X, k=3)

    # Display K-means clustering results
    st.subheader("K-means Clustering Results")
    st.write("Labels:", kmeans_labels)
    st.write("Centroids:", kmeans_centroids)

    # Display GMM clustering results
    st.subheader("Gaussian Mixture Model (GMM) Clustering Results")
    st.write("Labels:", gmm_labels)
    st.write("Means:", gmm_means)

if __name__ == "__main__":
    main()
