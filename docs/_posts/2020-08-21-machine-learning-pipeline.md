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

#### Data Imbalance

- Data imblanance
  - When working with classification models, however, there is something else to consider before moving forward: class balance.The number of occurrences of each class in the target variable is known as the class distribution. When predicting a categorical target, problems can arise when the class distribution is highly imbalanced. If there are not enough instances of certain outcomes, the resulting model might not be very good at predicting that class.Classification is a very broad field, with many applications across different industries. Some business needs, however, require a model to be good at classifying things that occur relatively rarely in the data. These types of problems have several names that are used commonly, including rare event prediction, extreme event prediction, and severe class imbalance. However, they all refer to the same thing: at least one of the classes in the target variable occurs much less frequently than another.
  - Class balancing 
    - refers to the process of changing the data by altering the number of samples in order to make the ratios of classes in the target variable less asymmetrical. It is a large field of study on its own, and there are several methods that allow you to balance the classes while maintaining the integrity of the data. Here, you’ll learn about some of the most common methods that can be used to create a better model.
      - Downsampling 
        - is the process of making the minority class represent a larger share of the whole dataset simply by removing observations from the majority class. It is mostly used with datasets that are large. But how large is large enough to consider downsampling? Tens of thousands is a good rule of thumb, but ultimately this needs to be validated by checking that model performance doesn’t deteriorate as you train with less data.
        - One way to downsample data is by selecting some observations randomly from the majority class and removing them from the dataset. There are some more technical, mathematically based methods, but random removal works very well in most cases.
      - Upsampling
        - Upsampling is basically the opposite of downsampling, and is done when the dataset doesn’t have a very large number of observations in the first place. Instead of removing observations from the majority class, you increase the number of observations in the minority class.
        - There are a couple of ways to go about this. The first and easiest method is to duplicate samples of the minority class. Depending on how many such observations you have compared to the majority class, you might have to duplicate each sample several times over.
        - Another way is to create synthetic, unique observations of the minority class. On the surface, there seems to be something wrong about editing the dataset like this, but if the goal is simply to train a better-performing model, it can be a valid and useful technique. You can generate these synthetic observations from the observations that currently exist. For example, you can average two points of the minority class and add the result to the dataset as a sample of the minority class. This can even be done algorithmically using publicly available Python packages.
      -  **it is important to leave a partition of test data that is unaltered by the sampling adjustment. 
        - You do this because you need to understand how well your model predicts on the actual class distribution observed in the world that your data represents. In the case of the spam detector example, it’s great if your model can score well on resampled data that is 80% not spam and 20% spam, but you need to know how it will work when deployed in the real world, where spam emails are much less frequent. This is why the test holdout data is not rebalanced.
      - Consequences
        - The first consequence is the risk of your model predicting the minority class more than it should. By class rebalancing to get your model to recognize the minority class, you might build a model that over-recognizes that class. That happens because, in training, it learned a data distribution that is not what it will be in the real world.
        - Changing the class distribution affects the underlying class probabilities learned by the model.
        - Class rebalancing should be reserved for situations where other alternatives have been exhausted and you still are not achieving satisfactory model results. Some guiding questions include:
          - How severe is the imbalance? A moderate (< 20%) imbalance may not require any rebalancing. An extreme imbalance (< 1%) would be a more likely candidate.
          - Have you already tried training a model using the true distribution? If the model doesn’t fit well due to very few samples in the minority class, then it could be worth rebalancing, but you won’t know unless you first try without rebalancing.
          - Do you need to use the model’s predicted class probabilities in a downstream process? If all you need is a class assignment, class rebalancing can be a very useful tool, but if you need to use your model’s output class probabilities in another downstream model or decision, then rebalancing can be a problem because it changes the underlying probabilities in the source data.


  - [imbalanced-learn](https://imbalanced-learn.org/stable/introduction.html): Introduction to imbalanced-learn, a library with tools to help with unbalanced datasets. Designed to work with scikit-learn.
  -[ RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html): imbalanced-learn documentation for a tool used to randomly upsample data
  - [Upsampling methods](https://imbalanced-learn.org/stable/references/over_sampling.html): imbalanced-learn documentation for various upsampling methods
  - [Downsampling methods](https://imbalanced-learn.org/stable/references/under_sampling.html): imbalanced-learn documentation for various downsampling metho

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


## Cross Validation

  - A rigorous approach to model development might use both cross-validation and validation. The cross-validation can be used to tune hyperparameters, while the separate validation set lets you compare the scores of different algorithms (e.g., logistic regression vs. Naive Bayes vs. decision tree) to select a champion model. Finally, the test set gives you a benchmark score for performance on new data. This process is illustrated in the diagram below.





