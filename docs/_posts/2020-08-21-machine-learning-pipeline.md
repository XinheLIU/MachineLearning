---
layout: page
title: "Machine Learning Pipeline"
date: 2020-08-21 15:00:00 -0000
categories: Classic
--- 

## Overview

Basic Problem solving with Machine Learning contains

- Gathering data.
- Cleaning data.
- Feature engineering.
- Defining model.
- Training, testing model and predicting the output.

## Data Collection



## Data Cleaing(Data Cleasing)

#### Handle Missing Values

- Common Methods - Analyze Impact
  - Deletion
  - Imputation
- Categorical Features
  - Make NA as one category
  - Logistic Regression \(Model\)
- Continuous Features
  - Mean, Median, Mode, etc
  - Linear Regression
    - iterative
    - introduces some degree or correlation among features, but can me remediated by separating into sub-groups

## Feature Engineering

### Feature Extraction

Imbalanced Features

- [Oversampling and Undersampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)
  - SMOTE, ADASYN
- Choose appropriate evaluation metrics
  
Sparse Features

- Tree-based model are bad with sparse features
- Use **Embeddings** to transfer higher dimensional data to lower dimensions
  - Tree based model + Logistic Regression/Neural Network

Scale and Transformation for Numerical Features

- standardization
- Normalization
- log transformation
- statistics (max, min, std)
- discretization
  - quantiles
  - bucketing

[Encoding](https://en.wikipedia.org/wiki/Character_encoding) for Categorical Features, Text features

- label encoding
  - cardinal to ordinal bias
- one-hot-encoding
  - frequency-based encoding
- target encoding
  - integrating label information to itself, easy to overfit
    - separate part of data solely used for encoding 
- too many features
  - frequency-based encoding, target-encoding
  - hashing or clustering to combine features
- too sparse features

Time-series

- Seasonality
  - month in a year, day in a week as features
- Relavant to events

##### Domain Knowledge based Methods

Text-data

- Word Frequency, Document Frequency
- N-Gram
- Word2Vec
- Weight of Evidence in Text

##### Information Theory base Methods

- Mutual Information
- Expected Cross Entropy
- QEMI
- Information Gain
- Odds Ratio

##### Model based Methods

- Genetic Algorithm
- Simulating Annealing
- PCA, LDA
- Language Models
  - LDA, LSA, pLSA

#### Feature Selections

- Filtering
  - on variance and correlation (Pearson, Spearman)
  - Mutual information
  - Feature importance
- Wrapper
  - Recursively deletion of feature space (on cross-validation)
    - stepwise feature selection
- Embedding
  - regularization
  - dimension reduction
    - PCA
    - LDA
