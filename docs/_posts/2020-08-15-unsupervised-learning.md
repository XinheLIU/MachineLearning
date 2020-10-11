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
      - [Isometric Mapping \(IsoMap\)](#isometric-mapping-isomap)
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

- Guarantee local optimum
  - Sensitive to the choice of cetners
- Hard to choose K

Extensions

- [K-Medoid](https://en.wikipedia.org/wiki/K-medoids)
- Customized Disimilarity and [distance](https://en.wikipedia.org/wiki/Distance)

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

features

- sensitive to E and minPts

pros and cons

- less sensitive to noise points
- density based (K-means is instance based)
- handle different shapes and size clusters better than K-Means
- less reliance on centroids
- no assumption on distribution of data
  - K-Means assumes normal mixtures
- stable result
  - multiple runs will have same resules

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
