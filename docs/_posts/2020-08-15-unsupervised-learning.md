---
layout: page
title: "Unsupervised Learning"
date: 2020-08-15 15:00:00 -0000
categories: Classic
--- 

- [Overview](#overview)
- [Clustering](#clustering)
  - [Distance](#distance)
  - [K-Means](#k-means)
    - [Pros and Cons of K-Means](#pros-and-cons-of-k-means)
  - [Hierarchical Clustering](#hierarchical-clustering)
  - [DBSCAN](#dbscan)
  - [OPTICS](#optics)
- [Dimension Reduction](#dimension-reduction)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [Multi-dimensional Scaling](#multi-dimensional-scaling)
  - [Non-Linear Dimension Reduction](#non-linear-dimension-reduction)
    - [Manifold Learning](#manifold-learning)
      - [Isometric Mapping (IsoMap)](#isometric-mapping-isomap)
    - [Locally Linear Embedding](#locally-linear-embedding)
- [Distance Metric Learning](#distance-metric-learning)
- [Association Rule Learning](#association-rule-learning)
  - [Aprori](#aprori)
  - [Eclat](#eclat)

## Overview

Unsupervised Learning are widely used in

- find relevant features
- compress information
- retrieve similar objects
- generate new data samples
- explore high-dimensional data

## Clustering

### [Distance](https://en.wikipedia.org/wiki/Distance)

- nonnegative, identity, symmetric
- subadditive \(triangle inequality\)
  
Examples

- [Euclidean Distance](https://en.wikipedia.org/wiki/Minkowski_distance)
- [Minkowski Distance](https://en.wikipedia.org/wiki/Minkowski_distance)
- Cosine Distance
- Manhattan Distance

### [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)

Model Training

- Use E-M Algorithm
- [hyperparameter K choice](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set)
  - Cross-Validation
    - Silhouette Coefficient
  - [calinski-harabasz index](https://stats.stackexchange.com/questions/97429/intuition-behind-the-calinski-harabasz-index)(C-H Index)

#### Pros and Cons of K-Means

K-means is a fundamental clustering algorithm with several important characteristics:

Pros:

- Simplicity: Easy to understand and implement, based on minimizing distances between points and centroids
- Scalability: Efficiently handles large datasets with many dimensions
- Speed: Fast convergence, especially with high dimensions
- Interpretability: Results are easy to visualize and explain

Cons:

- Initialization sensitivity: Final clusters depend heavily on initial centroid placement
- Requires pre-specified K: Number of clusters must be chosen in advance
- Outlier sensitivity: Outliers can significantly impact centroid calculations
- Spherical cluster bias: Assumes equal-sized, spherical clusters
  - Works best with round clusters by minimizing intercluster variance
- Scale dependency: Features must be normalized/standardized first
- Limited robustness: Struggles with uneven cluster sizes and varying densities

Evaluation Metrics:

- Inertia (elbow method)
- Silhouette analysis
  - [scikit-learn silhouette_score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html?highlight=silhouette#sklearn.metrics.silhouette_score)
  - [Original silhouette analysis paper](https://www.sciencedirect.com/science/article/pii/0377042787901257?via%3Dihub)

Important Extensions:

- K-means++: Improved initialization method
  - Default in scikit-learn
  - Uses probability-based centroid selection
  - Helps avoid local minima by spacing initial centroids
  - [Implementation details](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  - [Original K-means++ paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [K-Medoid](https://en.wikipedia.org/wiki/K-medoids): Variant using actual data points as centers
- Customized dissimilarity measures using different [distance metrics](https://en.wikipedia.org/wiki/Distance)

### Hierarchical Clustering

- top-down vs bottom-up \(agglomerative vs divisive\)
- linkage
  - complete, single
  - average
  - centroid
  - minimax
    - no inversion
    - nice interpreation
    - monotone transformation invariance
    - centers are chosen among data
- CH index
- Gaussian Mixture Model
- Expectation-Maximization
- Mean-Shift

### [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)

Algorithm

> - Available in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed together while marking points in low-density regions as outliers/noise.

The pseudocode below outlines the core DBSCAN algorithm:

1. The algorithm takes 3 inputs:
   - D: The dataset to be clustered
   - eps (ε): The radius that defines the neighborhood around each point
   - min_samples: Minimum points required in a neighborhood to form a dense region

2. Main algorithm flow:
   - Iterates through each unvisited point P in the dataset
   - For each point, finds all points within eps distance (its neighbors)
   - If point has fewer than min_samples neighbors, marks it as noise
   - Otherwise, starts a new cluster and expands it by recursively adding density-reachable points

3. Cluster expansion process (expand_cluster):
   - Adds the initial point to the current cluster
   - For each neighbor:
     - If unvisited, marks it as visited and checks its neighborhood
     - If neighbor has enough points in its neighborhood, adds those points to be checked
     - If neighbor isn't yet in a cluster, adds it to current cluster

The algorithm effectively identifies clusters of arbitrary shapes based on density, while automatically detecting and marking noise points that don't belong to any dense region.

```pseudocode
DBSCAN(D, eps, min_samples):
    # D: Dataset
    # eps (ε): Radius of neighborhood
    # min_samples: Minimum points required in ε-neighborhood
    
    C = 0  # Cluster counter
    for point P in D:
        if P is unvisited:
            mark P as visited
            neighbors = get_neighbors(P, eps)
            
            if len(neighbors) < min_samples:
                mark P as noise
            else:
                C = C + 1  # Start new cluster
                expand_cluster(P, neighbors, C, eps, min_samples)

expand_cluster(P, neighbors, C, eps, min_samples):
    add P to cluster C
    for point P' in neighbors:
        if P' is unvisited:
            mark P' as visited
            new_neighbors = get_neighbors(P', eps)
            if len(new_neighbors) >= min_samples:
                neighbors = neighbors ∪ new_neighbors
        if P' is not yet member of any cluster:
            add P' to cluster C
```

Hyperparameters

- **eps (ε)**: The radius that defines the neighborhood around each point
- **min_samples**: Minimum number of points required in the ε-neighborhood for a point to be considered a core point (including itself)

#### Pros and Cons of DBSCAN

Pros:

- Less sensitive to noise points compared to other clustering methods
- Density-based approach (unlike instance-based K-means)
- Better handling of clusters with different shapes and sizes
- Doesn't rely on centroids
- No assumptions about data distribution (unlike K-means which assumes normal mixtures)
- Stable results - multiple runs produce the same clustering

Cons:

- Sensitive to hyperparameters eps and min_samples
- Requires careful tuning for different datasets
- Not suitable for high-dimensional data

### Agglomerative Clustering

> - Available in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)

Agglomerative clustering is a hierarchical clustering algorithm that builds clusters by merging points/clusters in a bottom-up approach.

Algorithm:

1. Start with each point in its own cluster
2. Iteratively merge the two closest clusters until:
   - Desired number of clusters is reached, or 
   - Distance between clusters exceeds threshold
3. For each merge:
   - Calculate distances between all cluster pairs
   - Merge closest pair based on linkage criteria
   - Update distances to new merged cluster

Hyperparameters:

- **n_clusters**: Number of clusters to find (stopping criterion)
- **linkage**: Method to calculate inter-cluster distances
  - Single: Minimum distance between points in clusters
  - Complete: Maximum distance between points in clusters  
  - Average: Mean distance between points in clusters
  - Ward: Minimize within-cluster variance
- **affinity**: Distance metric between points (default: Euclidean)
- **distance_threshold**: Maximum distance for merging clusters

Pros:

- Hierarchical structure provides insights into relationships
- No assumptions about cluster shapes
- Can handle different distance metrics
- Deterministic results

Cons:

- Computationally expensive O(n^3)
- Sensitive to noise and outliers
- Cannot handle large datasets well
- Hard to determine optimal number of clusters

### [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm)

To overcome the sensitivity of DBSCAN to hyperparameters, only givesa reachability-plot (a special kind of dendrogram)

## Dimension Reduction

### [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA)

  - steps
    - normalization/centralization
    - eigen-value decomposition 
  - interpretation
    - maximize reconstructability -&gt; min distance
    - maximize separability 
  - extension
    - Kernel PCA
    - Probablistic PCA

### Multi-dimensional Scaling

  - from distance matrix to inner product matrix 

### [Non-Linear Dimension Reduction](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction)

#### Manifold Learning

##### [Isometric Mapping](https://en.wikipedia.org/wiki/Isomap) \(IsoMap\)

- build nearest neighbor map
- use MDS
- new data
  - need to build a regression model between high-low dimension 

#### [Locally Linear Embedding](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Locally-linear_embedding)
- t-SNE : t-distributed stochastic neighbor embedding

Supervised Learning methods such as Linear Discriminant Analysis could also be used

## Distance Metric Learning

- [Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  - weight for feature dimensions

## Association Rule Learning

### Aprori
### Eclat
