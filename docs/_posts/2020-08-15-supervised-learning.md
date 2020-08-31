---
layout: page
title: "Supervised Learning"
date: 2020-08-15 15:00:00 -0000
categories: Classic
---

## Concepts

* [Bias Variance Decomposition](https://en.wikipedia.org/wiki/Bias–variance_tradeoff)
  * Overfit
  * [Regularization](https://en.wikipedia.org/wiki/Regularization_%28mathematics%29)

## [Regression Models](https://en.wikipedia.org/wiki/Regression_analysis)

### [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)

* Intuition
  * Geometric Intepration of Least Squares
    * indeed a projection from **y** to **X** vector space
  * Likelibood and MLE
* [Assumptions](https://en.wikipedia.org/wiki/Linear_regression#Assumptions)
  * diagnostics and treatments
    * fitted and residual plot
    * heteroskedasticity
    * co-linearity
      * regularization via Lasso
      * PCA \(matrix transformation\)
* Analytical Solution
  * Single variable
    * $$\sum (y_i - (\beta_1 x_i + \beta_0))^2$$
    * xy vs yx regression \(notice whose variance get "contributed averaged"\)
  * Muti-variable 
    * $$\mathbf{w^*} = (\mathbf{X^T X})^{-1} \mathbf{X^T y} $$
  * [hat matrix](https://en.wikipedia.org/wiki/Projection_matrix)
* Regularization
  * Ridge
    * Thikhonov Regularization
  * Lasso (Sparcity)
    * Least Square Shrinkage and Selection
* [Robustness](https://en.wikipedia.org/wiki/Robust_regression#:~:text=In%20robust%20statistics%2C%20robust%20regression,variables%20and%20a%20dependent%20variable.)
  * Huber Loss Function
  * [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)
  * [Theil-Sen](/Theil–Sen estimator)
* [Bayesian Linear Regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression)

### Generative Linear Models

* Non-linear regression models
  * Nonparametric Regression
    * Complexity controlled by the smoothing parameter (bandwidth).
    * model complexity interpreted in *Degrees of Freedom/Effective degrees of freedom/equivalent degrees of freedom Residual Degrees of freedom* is n minus model degrees of freedom.
  * Local polynomial Regression
    * only fit a **neighborhood** of a target point. parameter $\alpha$ to control the span-traditionally, 0.5. When weighting the data in the neighborhood,
      Fit by weighted sum of squares
     * $$\sum_{i=1}^n w_i (y_i  - (\beta_0 + \beta_1 x_i))^2$$
       * $$w_i = = \left\{
                \begin{array}{lr}
                (1-|\frac{x_i-x_0}{\text{max dist}}|^3)^3 \text{ ,if $x_i$ is in the neighborhood} &  \\
                0, \text{ otherwise } &  
                \end{array}
         \right.$$
  
  * [Splines](https://en.wikipedia.org/wiki/Spline_%28mathematics%29)
    * Penalized (Smoothing) Splines: find twice differentiable x to minimize 
      * $$\sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int [ f^{(2)}(x)]^2 dx $$
        * $\lambda$ penalty for wiggy function. search of x can be a combination o f **basis functions** (n + 4 basis functions, n is the knots)
    * Cubic Splines
* Generalized Additive Models

## Classification

### Model Evaluation

#### [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

* Accuracy $$\int_{x\in D} \mathbb{I}(f(x)\neq y) p(x)dx$$
* Precision, Positive Predictive Value \(PPV\)
  * $$PPV =\frac{TP}{TP+FP}$$
  * $$P(H_1|x>\tau_{\alpha})$$
* Recall, Sensitivity, True Positive Rate, Power of the Test
  * $$TPR = \frac{TP}{TP+FN}$$
  * $$1- \beta= P(x \geq \tau_{\alpha} \vert H_1)$$
* False Positive Rate, Type II error Probability
  * $$FPR = \frac{FP}{FP+TN}$$
  * $$\beta= P(x\leq\tau_{\alpha})|H_1)$$
* False Negative Rate = Type I Error probability, Significance of the Test
  * $$FNR = \frac{FN}{FN+TP}$$
  * $$\alpha=
    P(x \leq\tau_{\alpha})|H_0)$$
* Specificity, Selectivity, True Negative Rate
  * $$TNR = \frac{TN}{TN+FP}$$
* False Discover Rate
  * $$FDR = \frac{TN}{TN+FP}$$
  * $$P(H_0|x>\tau_{\alpha})$$
* P-R Curve
* $F-\beta$ Measure
  * $$\frac{1}{F_{\beta}} = \frac{1}{1+\beta^2}(\frac{1}{P}+\frac{\beta^2}{R})$$
  * $$F_{\beta} = \frac{(1+\beta^2) \times P \times R}{(\beta^2 \times P) + R}$$
* ROC \(Receiver Operating Characteristic\) Curve
  * AUC \(Area Under Curve\)
    * $$AUC = \int_{-\infty}^{+\infty} TPR(t)(FPR(t))'dt= \int_{-\infty}^{+\infty} \int_{t}^{+\infty} f_1(x)dxf_0(t)dt= \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} \mathbb{1}_{x>T} f_1(x) f_0(t)dxdt$$
    * $$AUC= \mathbb{P}(S_1 > S_0) = 1-Loss_{rank}$$
* Cost-Sensitive loss:  with unequal loss to FP and FN
  * Cost Curve: use to measure cost-sensitive error rate, P+ cost as horizontal and normalized cost as vertical

#### Other methods

[Calibration curve](https://en.wikipedia.org/wiki/Calibration_curve)

### Models

* [Generative Models](https://en.wikipedia.org/wiki/Generative_model)
* [Discriminative Models](https://en.wikipedia.org/wiki/Discriminative_model)

#### [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression)

1. Intuition - Why do we have it

   * different contribution of large/small data
     * exponential family and penalize
   * assumption - Bernoulli distribution
     * odds, log odds

2. Loss function derive parameter estimation

   * MLE - y is bernoulli function \(entropy loss\)

3. implement Gradient Descent

   * entropy loss \(Bernoulli MLE Loss\)
   * Mean Square Loss

#### Tree-Based Models

##### [Tree](https://en.wikipedia.org/wiki/Decision_tree_learning)

* 3-steps
  * feature selection
  * tree generation
  * prune
* Algo
  * ID3
    * based on absolute information gain
      * bias towards rare cases
  * ID4.5
    * based on[ information gain ratio](https://en.wikipedia.org/wiki/Information_gain_ratio)
  * CART
    * based on [Gini-coefficient](https://en.wikipedia.org/wiki/Gini_coefficient)
  * Hyperparamters
    * max\__depth, min\_\_samples\_\_split, min\_samples\_leaf, max\_leaf\_nodes\_
* Advantages
  * interaction handling
  * insensitivity to outliers
* Disadvantage
  * feature dominance - high bias

##### Tree Ensembling

[Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)

* Random Forest
  * [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
    * bagging of data 
    * bagging of feature 
      * rule of sum - select sqrt\(k\) features
  * feature importance calculation
    * Mean Decrease Accuracy \([MDA](https://stats.stackexchange.com/questions/197827/how-to-interpret-mean-decrease-in-accuracy-and-mean-decrease-gini-in-random-fore)\)
    * out-of-bag performance
  * Advantages and Disadvantages
    * natural parallel \(embarrassed parallel algo\)
    * feature importance
* Gradient Boosting Tree
  * [Boosting](https://en.wikipedia.org/wiki/Boosting_%28machine_learning%29)
  * [Adaboost](https://en.wikipedia.org/wiki/AdaBoost)
    * loss function: exponential loss
      * other choices are sigmoid loss, hinge loss, log loss, etc
    * shrinkage
    * subsampling
  * [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
    * XGBoost
    * Catboost

#### Support Vector Machine

[SVM](https://en.wikipedia.org/wiki/Support_vector_machine) a Linear Discriminative Model

* Hinge Loss
* [Kernel Method](https://en.wikipedia.org/wiki/Kernel_method)
  * linear kernel
  * polynomial kernel
  * radial kernel

#### K-nearest Neighbor

* No Training stage
* higher  K is, more robust the model can be

#### Linear Discriminative Analysis

[LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) and QDA \(Quadratic Discriminative Analysis\) are generative models

* [Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  * Data Sphering
* Generalized Rayleigh Quotient

#### Naive Bayes Model

[Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

* Assume independence of features
* [Laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) / Laplacian correction
* Bayesian Method
* Lazy evaluation
* Semi-naive Bayesian Classifiers
  * one-dependent estimator
    * super-parent ODE
    * Tree Augmented Naive Bayes
    * Average One-dependent ODE