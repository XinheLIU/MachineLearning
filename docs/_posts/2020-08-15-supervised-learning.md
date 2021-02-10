---
layout: page
title: "Supervised Learning"
date: 2020-08-15 15:00:00 -0000
categories: Classic
---
- [Concepts](#concepts)
  - [Bias Variance Decomposition](#bias-variance-decomposition)
  - [Cross-Validation)](#cross-validation)
- [Regression](#regression)
  - [Linear Regression](#linear-regression)
    - [Assumptions](#assumptions)
    - [Linear Regression Model](#linear-regression-model)
      - [Least Square Approximation](#least-square-approximation)
      - [Multivariate Case](#multivariate-case)
      - [MLE(Maximum Likelihood) Solution](#mlemaximum-likelihood-solution)
      - [Prediction Confidence](#prediction-confidence)
    - [Testing of Assumptions](#testing-of-assumptions)
    - [Resolutions of Assumption Violations](#resolutions-of-assumption-violations)
    - [Model Selection](#model-selection)
      - [Regularization, Ridge, Lasso](#regularization-ridge-lasso)
        - [Ridge](#ridge)
        - [Lasso](#lasso)
  - [Bayesian Linear Regression](#bayesian-linear-regression)
  - [Generative Linear Models](#generative-linear-models)
- [Classification](#classification)
  - [Model Evaluation](#model-evaluation)
    - [Confusion Matrix](#confusion-matrix)
    - [Other methods](#other-methods)
  - [Models](#models)
    - [Logistic Regression](#logistic-regression)
      - [Intuition](#intuition)
      - [Loss Function](#loss-function)
      - [Model Training - Gradient Descent](#model-training---gradient-descent)
        - [Regularization of Entropy Loss](#regularization-of-entropy-loss)
      - [Multi-class classification](#multi-class-classification)
        - [Softmax](#softmax)
      - [Pros and Cons of Logistic Regression](#pros-and-cons-of-logistic-regression)
        - [Bayesian Intepretation and Linkage with Naive Bayes](#bayesian-intepretation-and-linkage-with-naive-bayes)
    - [Tree-Based Models](#tree-based-models)
      - [Decision Tree](#decision-tree)
      - [Tree Ensembling](#tree-ensembling)
        - [Random Forest](#random-forest)
        - [Boosting Trees](#boosting-trees)
    - [Support Vector Machine](#support-vector-machine)
      - [Intuition of SVM](#intuition-of-svm)
      - [Hinge Loss](#hinge-loss)
      - [Kernel Method](#kernel-method)
      - [Soft Margin, Slack Variable and Regularization](#soft-margin-slack-variable-and-regularization)
      - [Pros and Cons of SVM](#pros-and-cons-of-svm)
        - [Choose  kernels](#choose--kernels)
    - [Naive Bayes](#naive-bayes)
      - [Assumptions and Intuition of Naive Bayes](#assumptions-and-intuition-of-naive-bayes)
        - [The Independence Assumption](#the-independence-assumption)
        - [Lapalcian Correction](#lapalcian-correction)
      - [Semi-naive Bayesian Classifiers](#semi-naive-bayesian-classifiers)
      - [Pros and Cons of Naive Bayes](#pros-and-cons-of-naive-bayes)
    - [K-nearest Neighbor](#k-nearest-neighbor)
    - [Linear Discriminative Analysis](#linear-discriminative-analysis)
  
## Concepts

### [Bias Variance Decomposition](https://en.wikipedia.org/wiki/Bias–variance_tradeoff)

- Overfit
- [Regularization](https://en.wikipedia.org/wiki/Regularization_%28mathematics%29)

### [Cross-Validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

Out-of-sample testing, particularly useful for hyperparameter tuning in Machine Learning set-ups.

- hold-one out vs k-fold
  - information leaking when use to validate parameters
- Use Bootstrapping
  - 1/e rule
- Time-Series Data
  - not use future information
  - ARIMA
- Decompose bias and variance
  - in-sample vs out-of sample error

## [Regression](https://en.wikipedia.org/wiki/Regression_analysis)

### [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)

Intuition - based on Least Sqaure Distance

- Geometric Intepration of Least Squares
  - indeed a projection from **y*- to **X*- vector space
- Likelibood and MLE
  
#### [Assumptions](https://en.wikipedia.org/wiki/Linear_regression#Assumptions)

- Linear Relationship between covariates and dependent variable
- $E(\varepsilon)=0$
- $Var(\varepsilon) =\sigma^2$: Homoscedasticity
- $\varepsilon$ is independent with covariates 
- x is observed without error (and no perfect multicollinearity in multivariate case)
- (optional, Gauss-Markov Theorem) $\varepsilon$ is normal - when it is, OLS and MLE agrees and to be BLUE(Best Linear Unbiased Estimator)

#### Linear Regression Model

##### Least Square Approximation

Under Normal Condition, we have
$$y \sim N(\beta_0 + \beta x_i, \sigma^2)$$
$$ L(\theta) = (\frac{1}{\sqrt{2\pi} \sigma})^n exp( - \frac{\sum_{i=1}^n (y_i - (\beta_0 +\beta_1x_i))^2}{2\sigma^2})$$

Equivalent to minimize

$$RSS(\theta) = \sum_{i=1}^n (y_i - (\beta_0 +\beta_1x_i))^2$$
$$\partial_{\beta_i} RSS = 0, i =0,1$$
, we get
$$ r_{xy} = \frac{s_{xy}}{s_xs_y}, \beta_1 = r_{xy}\frac{s_y}{s_x} =\frac{s_{xy}}{s_x^2},\beta_0 = \bar{y}-\hat{\beta}\bar{x} $$

##### Multivariate Case

In Multi-variate Case:

Algebra approach

$$ f(x) = \mathbf{w}^T \mathbf{x} = \sum_{i=1}^n w_i x_i $$

$$\mathbf{w^*} = argmin_{\mathbf{\hat{w}}}(\mathbf{y}-\mathbf{X \hat{w}} )^T(\mathbf{y}-\mathbf{X \hat{w}} )$$

$$\frac{\partial{E}}{\partial{\mathbf{\hat{w}}}} = 2 \mathbf{X}^T(\mathbf{X \hat{w}} - \mathbf{y})$$
$$ \mathbf{w^*} = (\mathbf{X^T X})^{-1} \mathbf{X^T y} $$

Geometry Approach

$\mathbf{e} = \mathbf{b} - \mathbf{Ax}$ has no solution because there are too many equations. So we try

$$A^TA\mathbf{\hat{x}} = A^Tb$$

This is indeed a projection of b to column space of A.

$$A^T(b-A\mathbf{\hat{x}}) = 0$$

$$p=\hat{x_1}\mathbf{a_1} + \hat{x_2}\mathbf{a_2} + ...\hat{x_n}\mathbf{a_n}= A\mathbf{\hat{x}}$$

is the projection. Plug in $\mathbf{\hat{x}}$

$$P = \mathbf{\hat{x}} = A(A^TA)^{-1}A^T$$

[Hat Matrix](https://en.wikipedia.org/wiki/Projection_matrix): The relationship of predicted value and response

$$Y = H\hat{Y}$$
$$H = X(X^TX)^{-1}X^T$$

The Diagonal Entires $h_{ii}$ are the **Leverages.**

##### MLE(Maximum Likelihood) Solution

Assuming noise is normal, maximize

$$p(\mathbf{x_1, x_2 ... x_n}| \mathbf{ w} ) = \prod_k \frac{1}{\sqrt{2\pi} \sigma} exp[ -\frac{1}{2\sigma^2} (y_k - \mathbf{w_t x_k} )^2 ]$$ 

Another matrix representation

$$f(\beta) = min (Y-X\beta)^T(Y-X\beta), f'(\beta) = 2X^T(Y-X\hat{\beta}) = 0$$ 

to solve $\hat{\beta}$

$$ min ||y_k - \mathbf{w^T x}_k ||^2 + \lambda ||\mathbf{w}||_1 $$

##### Prediction Confidence

Variance Error In Prediction

$$V(\hat{y^*} - y^*) = \sigma^2 + \sigma^2[\frac{1}{n} + \frac{x^*-\bar{x})^2}{(n-1)s_x^2}]$$
$$ = V(E(y^*) - y^*) + V(\hat{y^*} - E(y^*)) + 2 cov(\hat{y^*} - y^*, \hat{y^*} - y^*)$$

The cross term is zero, the first term is variance with $\varepsilon^*$, second term is variance in $\beta$.

The confidence interval is $\hat{y^*} \pm t_{\alpha/2, n-2} SE(\hat{y^*})$. 

$R^2$, the coefficient of determination: The proportion of the sum of squared response which is accounted by the model relative to the model with no covariance. (take mean of response)

$$R^2 = 1- \frac{\sum(y_i-\hat{y_i})^2}{\sum(y_i-\bar{y})}$$

Note that $0 \leq R^2 \leq 1$ It only tells predictive power if the model is a good fit.

Adjusted $R^2$: $R^2$ + penalty P


#### Testing of Assumptions

- Scatter Plot 
  - check Linear Relationship and Outliers
- Residual Analysis ($\hat{\varepsilon} = y - \hat{y}$)  
  - Diagnostic Plots:
    - Plot of Residuals vs. Fitted Values
    - Normal Probability Plot 
    - Plot Residuals versus time (see any trend of fit)
- Cook's Distance 
  - $$D_j = \frac{\sum_{i=1}^n (\hat{y_i} - \hat{y}_{i(-j)})^2}{(p+1)\hat{\sigma}^2}$$ 
    - Test Against $F_{(p+1),(n-p-1}$ degrees of freedom, over 50th percentile will definitely become a problem
- Detect Multicollinearity (two or more predictors are strongly related to one) 
  - Use **Variance Inflation Factor**
  - $$VIF_k = \frac{1}{1-R_k^2}$$
    - fit feature k against other predictors. Note VIF does not give any information of specific predictors

#### Resolutions of Assumption Violations

- Linear Relationship
  - non-linear regression, generalized linear models
  - Transformations ( for outliers, heteroskesticities, etc)
  - Use different models on different periods/data
- Outliers and Heteroskesticity
  - Weighted Least Squares regression,  (for outliers, heteroskesticity) 
  - [Robust Regression](https://en.wikipedia.org/wiki/Robust_regression#:~:text=In%20robust%20statistics%2C%20robust%20regression,variables%20and%20a%20dependent%20variable.)
    - $$\sum_{i=1}^n \rho(\frac{y_i-x_i^T\beta}{\sigma})$$
    - **Huber Loss Function**
      - $$\rho(x) = \left\{
            \begin{array}{lr}
            x^2 \text{  ,if } |x|<k &  \\
            k(2(|x|-k), \text{ otherwise } &  
            \end{array}
        \right.$$
      - (default k=1.345) (when k=0, it is an L1-regression, $K\to \infty$, the regression goes back to a linear regression model. It is effective in down-weighting the extreme examples.
    - [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)
    - [Theil-Sen](/Theil–Sen estimator)
- Colinearity
  - Sparsity - regularization via Lasso
  - PCA \(matrix transformation\)
- Inputs are discrete 
  - Factor Inputs (discrete features)
    - a factor of k levels adds k-1 terms into the regression function.(k-1 different $beta$s)
  
#### Model Selection

- Exhaustive Search by AIC or BIC (more stable than LOOCV)
- **Stepwise Regression/Stepwise Variable Selection** (At each step one covariate is added or dropped)
- Cross-Validation 
  - Leave-one-Out cross Validation of Linear Regression: Prediction Error Sum of Squares 
    - $$PRESS = \frac{\sum(y_i-\hat{y}_{-i})^2}{n}$$
    - $$y_i-\hat{y}_{-i} = \frac{\hat{\varepsilon}_i}{1-h_{ii}}$$ 
      - h is the leverage (hat matrix)

##### Regularization, Ridge, Lasso

###### Ridge

Thikhonov Regularization

$$Rss + \lambda \sum_{i=1}^p \beta^2$$

$\lambda$ is the regularization parameter. The result of Ridge is a \textbf{Shrinkage} of $\hat{\beta}$ towards zero.

Note

- No penalty for $\beta+0$ or b.
- The predictors should usually be standardized prior to fitting
- Choose $\lambda$ by cross-validation

###### Lasso

Lasso(Least Absolute Shrinkage and Selection Operator)

$$Rss + \lambda \sum_{i=1}^p |\beta|$$

Can be extended to 
$$-log likelihood + \lambda \sum_{i=1}^p |\beta|$$

Extensions

- Group Lasso
  - group predictors together to be either included or excluded. 
- [Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization)
  - $$Rss + \lambda \sum_{i=1}^p (\alpha|\beta_j| + (1-\alpha) \beta_j^2)$$

### [Bayesian Linear Regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression)

### Generative Linear Models

- Non-linear regression models
  - Nonparametric Regression
    - Complexity controlled by the smoothing parameter (bandwidth).
    - model complexity interpreted in *Degrees of Freedom/Effective degrees of freedom/equivalent degrees of freedom Residual Degrees of freedom- is n minus model degrees of freedom.
  - Local polynomial Regression
    - only fit a **neighborhood*- of a target point. parameter $\alpha$ to control the span-traditionally, 0.5. When weighting the data in the neighborhood,
      Fit by weighted sum of squares
     - $$\sum_{i=1}^n w_i (y_i  - (\beta_0 + \beta_1 x_i))^2$$
       - $$w_i = = \left\{
                \begin{array}{lr}
                (1-|\frac{x_i-x_0}{\text{max dist}}|^3)^3 \text{ ,if $x_i$ is in the neighborhood} &  \\
                0, \text{ otherwise } &  
                \end{array}
         \right.$$
  
  - [Splines](https://en.wikipedia.org/wiki/Spline_%28mathematics%29)
    - Penalized (Smoothing) Splines: find twice differentiable x to minimize 
      - $$\sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int [ f^{(2)}(x)]^2 dx $$
        - $\lambda$ penalty for wiggy function. search of x can be a combination o f **basis functions*- (n + 4 basis functions, n is the knots)
    - Cubic Splines
- Generalized Additive Models

## Classification

### Model Evaluation

#### [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

- Accuracy 
  - $$\int_{x\in D} \mathbb{I}(f(x)\neq y) p(x)dx$$
- Precision, Positive Predictive Value \(PPV\)
  - $$PPV =\frac{TP}{TP+FP}$$
  - $$P(H_1|x>\tau_{\alpha})$$
- Recall, Sensitivity, True Positive Rate, Power of the Test
  - $$TPR = \frac{TP}{TP+FN}$$
  - $$1- \beta= P(x \geq \tau_{\alpha} \vert H_1)$$
- False Positive Rate, Type I error Probability
  - $$FPR = \frac{FP}{FP+TN}$$
  - $$\alpha=
    P(x \leq\tau_{\alpha}|H_0)$$
- False Negative Rate = Type II Error probability, Significance of the Test
  - $$FNR = \frac{FN}{FN+TP}$$
  - $$\beta= P(x\leq\tau_{\alpha}|H_1)$$
- Specificity, Selectivity, True Negative Rate
  - $$TNR = \frac{TN}{TN+FP}$$
- False Discover Rate
  - $$FDR = \frac{TN}{TN+FP}$$
  - $$P(H_0|x>\tau_{\alpha})$$
- P-R Curve
- $F-\beta$ Measure
  - $$\frac{1}{F_{\beta}} = \frac{1}{1+\beta^2}(\frac{1}{P}+\frac{\beta^2}{R})$$
  - $$F_{\beta} = \frac{(1+\beta^2) \times P \times R}{(\beta^2 \times P) + R}$$
- ROC \(Receiver Operating Characteristic\) Curve
  - AUC \(Area Under Curve\)
    - $$AUC = \int_{-\infty}^{+\infty} TPR(t)(FPR(t))'dt= \int_{-\infty}^{+\infty} \int_{t}^{+\infty} f_1(x)dxf_0(t)dt= \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} \mathbb{1}_{x>T} f_1(x) f_0(t)dxdt$$
    - $$AUC= \mathbb{P}(S_1 > S_0) = 1-Loss_{rank}$$
- Cost-Sensitive loss:  with unequal loss to FP and FN
  - Cost Curve: use to measure cost-sensitive error rate, P+ cost as horizontal and normalized cost as vertical

#### Other methods

[Calibration curve](https://en.wikipedia.org/wiki/Calibration_curve)

### Models

- [Generative Models](https://en.wikipedia.org/wiki/Generative_model)
  - Bayesian based models: Naive bayesian, HMM, Bayesian Nets, Markov Random Fields
  - Mixure Models: GMM
- [Discriminative Models](https://en.wikipedia.org/wiki/Discriminative_model)
  - Logistic Regression, SVM, Tree-based models
  - Neural Nets

---
#### [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression)

##### Intuition

1. Sigmoid/ Log Probability Function (linkage function) - Logistic Regression is a type of [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) with sigmoid linkage

$$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}$$

- different contribution of large/small data
  - exponential family and penalize

2. Assume y is **Bernoulli distributed** and  $P(Y|X) = \sigma(X)$ (conditional probability plot), indeed we have **odds**

$$\text{logit } p = log(\frac{p}{1-p}) = \mathbf{w}^T\mathbf{x}$$

(log odds, logit function)

$$\frac{p}{1-p} = e^{\mathbf{w}^T\mathbf{x}}$$
$$ p(y=1|x) = \frac{e^{\mathbf{w}^T\mathbf{x}}}{1+e^{\mathbf{w}^T\mathbf{x}}} = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}$$
$$p(y=0|x) = \frac{1}{1+e^{\mathbf{w}^T\mathbf{x}}}$$

3. a more general inituition is the [Principle of Maximum Entropy](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy) (see Loss function) of the exponential family

$$2 \sum_{i=1}^N -y_i log\frac{y_i}{\hat{p_i}}+ (1-y_i)log(\frac{1-y_i}{1-\hat{p_i}})$$

##### Loss Function

$$J(z) = -ylogy + (1-y)log(1-y)$$

MLE of w based on a Bernoulli Distribution

$$l(\mathbf{w}|\mathbf{x}) = L(\mathbf{w}|\mathbf{x}) = \prod_{i=1}^N[p(y=1|\mathbf{x,w})]^{y_i}[1-p(y=1|\mathbf{x,w})]^{1-y_i}$$
Log likelihood
$$log L(\mathbf{w}|\mathbf{x}) = \sum_{i=1}^N -y_i log p_i + (1-y_i)log(1-p_i)$$

##### Model Training - Gradient Descent

$$
\frac{\partial l }{\partial \mathbf{w}} = \frac{\partial l }{\partial p_i}
        \frac{\partial p_i }{\partial z_i} \frac{\partial z_i }{\partial \mathbf{w}} = 
        (\frac{y_i}{p_i} - \frac{1-y_i}{1-p_i})(p_i(1-p_i))(\mathbf{x})
$$

$$w_{j} := w_j + \alpha (y^{(i)} - \sigma_{w_i}(x^{(i)}_j )) x^{(i)}$$

Here we used the **derivative of sigmoid**

$$ \frac{\partial \sigma(z)}{\partial z} = \frac{\partial p }{\partial z} 
  = \frac{-e^{-z}}{(1+e^{-z})^2}
  = \sigma(z)(1-\sigma(z))$$

###### Regularization of Entropy Loss

With L-1 or L-2 norm (Frobenius Norm)

$$J(z) = \frac{1}{m} \sum_{i=1}^m L(y_i,\hat{y}_i) + \frac{\lambda}{2m}\|{\mathbf{w}}\|^2_F$$

##### Multi-class classification

Use multiple binary classifiers

- One-vs-One(OvO): $\frac{N(N-1)}{2}$ two-class classifications: Then vote among classification results
- One-vs-Rest(OvR): one as positive, consider all rest negative: combine the results 
- Many-vs-Many(MvM): Cut N Samples with M partitions, do classification on M training examples. Use Error Correction Output Codes(EOOC) to get minimum-distance ones as result

###### [Softmax](https://en.wikipedia.org/wiki/Softmax_function)

$$P(Y=k|x) = \frac{e^{\mathbf{w}^T\mathbf{x}}}{\sum_{i=1}^{K}e^{\mathbf{w}^T\mathbf{x}}}$$

notice, $\mathbf{w} = (\theta_1, ..., \theta_n)$ has redundancy, when K = 2, it is logistic regression

##### Pros and Cons of Logistic Regression

Usually serve as a base line model because of nice intepretation and Principle of Maximum Entropy

###### Bayesian Intepretation and Linkage with Naive Bayes

- Naive Bayesian assumes $p(x_i|Y=y_k)$ follows a normal distribution. Then the posterior probability is
  - $$ P(Y=0|x) = \frac{P(Y=0)P(X|Y=0)}{P(Y=0)P(X|Y=0) + P(Y=0)P(X|Y=1)} $$
  - $$= \frac{1}{1+ exp(ln\frac{P(Y=0)P(X|Y=1)}{P(Y=0)P(X|Y=0)})} $$
  - $$ = \frac{1}{exp(ln\frac{1-p_0}{p_0} + \sum(\frac{\mu_{i1}-\mu_{i0}}{\sigma_i^2}X_i + \frac{\mu_{i0}^2-\mu_{i1}^2}{2\sigma_i^2}))}$$
- Though the solution follows the exact same pattern, Logistic Regression does not have the assumption of independence. When assumptions differ, the results differ. Generally, logistic regression results less bias, more variance(more flexible)
- The rate of convergence is also different, logistic regression needs more data feeding to perform better.
  
---

#### Tree-Based Models

##### [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning)

- 3-steps
  - feature selection
  - tree generation
  - prune
- Algo
  - ID3
    - based on absolute information gain
      - bias towards rare cases
  - C4.5
    - based on[information gain ratio](https://en.wikipedia.org/wiki/Information_gain_ratio)
  - CART
    - based on [Gini-coefficient](https://en.wikipedia.org/wiki/Gini_coefficient)
  - Hyperparamters
    - max__depth, min_samples_split, min_samples_leaf, max_leaf_nodes
- Advantages
  - interaction handling
  - insensitivity to outliers
- Disadvantage
  - feature dominance - high bias

##### Tree Ensembling

[Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)

###### Random Forest

- [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
  - bagging of data 
  - bagging of feature 
    - rule of sum - select sqrt\(k\) features
- feature importance calculation
  - Mean Decrease Accuracy \([MDA](https://stats.stackexchange.com/questions/197827/how-to-interpret-mean-decrease-in-accuracy-and-mean-decrease-gini-in-random-fore)\)
  - out-of-bag performance
- Advantages and Disadvantages
  - natural parallel (embarrassed parallel algo)
  - feature importance

###### Boosting Trees

- [Boosting](https://en.wikipedia.org/wiki/Boosting_%28machine_learning%29)
- [Adaboost](https://en.wikipedia.org/wiki/AdaBoost)
  - loss function: exponential loss
    - other choices are sigmoid loss, hinge loss, log loss, etc
  - shrinkage
  - subsampling
- [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
  - XGBoost
  - Catboost
  - LightGBM
    - faster, histogram based
    - ligher, bins for continuous data
    - accuracy boosting
    - leaf-wise split
      - other methods split level-wise
      - generate more complex trees
    - supports parallelization
  
---

#### [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)

##### Intuition of SVM

Find an hyperplane can separate all the samples:

$$\left\{
         \begin{array}{lr}
             \mathbf{w}^T\mathbf{x} + b \geq +1,  y = +1 &  \\
             \mathbf{w}^T\mathbf{x} + b \leq -1,  y = -1  
             \\
             \end{array}
\right.$$

The vectors make "=" are the support vectors.

The margin is

$$\gamma = \frac{2}{\|\mathbf{w}\|}$$ 
($\frac{\mathbf{w}^T\mathbf{x} + b}{\|\mathbf{w}\|}$ is the point distance to plane)

So the problem is
$$ argmax_{\mathbf{w},b} \frac{2}{\|\mathbf{w}\|}$$
$$s.t. (\mathbf{w}^T\mathbf{x}_i + b)y_i \geq 1$$

Equivalent to

$$ argmin_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2$$
$$s.t. (\mathbf{w}^T\mathbf{x}_i + b)y_i \geq 1$$

Lagrange Multiplier

$$L = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^m\alpha_i(1 - (\mathbf{w}^T\mathbf{x}_i + b)y_i)$$

with first-order condition for $\mathbf{w}$ and b we can have

$$\mathbf{w} = \sum_{i=1}^m \alpha_i y_i \mathbf{x}_i, b = \sum_{i=1}^m \alpha_i y_i$$

Then we get the [dual problem](https://en.wikipedia.org/wiki/Duality_(optimization))

$$argmax_{\boldsymbol{\alpha}}  \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$
$$s.t \sum_{i=1}^m \alpha_i y_i = 0$$
$$ \alpha_i \geq 0$$

Notice, the dual problem

- primal problem is related to feature dimentions, dual problem not
- directly related to kernel representation

When Satisfy [K.K.T (Karush-Kuhn-Tucker) condition](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)

$$\left\{
         \begin{array}{lr}
             \alpha_i \geq +1,  y = +1 &  \\
              y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \geq 0 \\
              \alpha_i( y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 ) \geq 0 
             \end{array}
   \right.$$ 

(See Optimization), could be solved using SMO([Sequential Minimal Optimization](https://en.wikipedia.org/wiki/Sequential_minimal_optimization))

##### Hinge Loss

The loss of SVM is a [Hinge Loss](https://en.wikipedia.org/wiki/Hinge_loss) function

$$\sum_i^m [1-y_i(\mathbf{w} x_i + b)]_+ \lambda \|w\|^2$$

##### [Kernel Method](https://en.wikipedia.org/wiki/Kernel_method)

For Linear Un-separable problems, we can project to higher-dimensions

$$ argmax_{\mathbf{w},b} \frac{2}{\|\mathbf{w}||}$$
$$s.t. (\mathbf{w}^T\phi(\mathbf{x}_i)+ b)y_i \geq 1$$

$$argmax_{\boldsymbol{\alpha}}  \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$$
$$s.t \sum_{i=1}^m \alpha_i y_i = 0$$
$$ \alpha_i \geq 0$$

Kernel

$$\kappa(\mathbf{x}_i, \mathbf{x}_j ) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$$

We can find the solution by

$$f(x) = (\mathbf{w}^T\phi(\mathbf{x}_i)+ b) =\sum_{i=1}^m \alpha_i y_i 
\kappa(\mathbf{x}, \mathbf{x}_i ) + b$$

Theorem:

When a symmetric function has semi-positive definite kernel matrix

$$\begin{pmatrix} 
    \kappa(\mathbf{x}_1, \mathbf{x}_1 )& \cdots & \kappa(\mathbf{x}_1, \mathbf{x}_m )\\ 
    \vdots & \ddots & \vdots \\
    \kappa(\mathbf{x}_m, \mathbf{x}_1 )& \cdots & \kappa(\mathbf{x}_m, \mathbf{x}_m )  
\end{pmatrix} $$

it can be a kernel function.

Common Kernels are

- Linear Kernel
  - $$\kappa(\mathbf{x}_1, \mathbf{y} ) = \mathbf{x}^T\mathbf{y}$$
- Polynomial Kernel 
  - $$\kappa(\mathbf{x}_1, \mathbf{y} ) = (\mathbf{x}^T\mathbf{y} + c)^d$$
- Gaussian Kernel 
  - $$\kappa(\mathbf{x}_1, \mathbf{y} ) = exp(-\frac{\|\mathbf{x}-\mathbf{y}^2}{2\sigma^2})$$
- Laplace Kernel
  - $$\kappa(\mathbf{x}_1, \mathbf{y} ) = exp(-\frac{\|\mathbf{x}-\mathbf{y}\|}{\sigma})$$
- sigmoid Kernel
  - $$\kappa(\mathbf{x}_1, \mathbf{y} ) = tanh(\beta \mathbf{x}^T\mathbf{y} + \theta)$$

##### Soft Margin, Slack Variable and Regularization

Introduce Slace varaible $\xi$ to allow soft margin.

$$argmin_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^m \xi_i$$
$$s.t. (\mathbf{w}^T\mathbf{x}_i + b)y_i \geq 1 - \xi_i$$
$$\xi_i \geq 0$$

##### Pros and Cons of SVM

- Only considers support vector (maxizes difference)
  - **Less overfitting**
- More complex than logistic regression
- Handles high-dimensional data better
- Sensitive to imblanaced samples

###### Choose  kernels

- linear kernel if linearly separable
  - with large, seprable samples
- Gaussian kernel maps to infinty dimension spaces
  - not too much feature and samples, not very time-sensitive
- RBF Kernel
  - more hyperparameters, more complexity
- Can use expert prior
- Use cross-validation
- Mixed kernels - take weighted average

---

#### [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

##### Assumptions and Intuition of Naive Bayes

When we have a 0-1 loss

$$L = \left\{
             \begin{array}{lr}
             0 \text{ , if i=j} &  \\
             1, \text{ otherwise } &  
             \end{array}
      \right.$$

the risk become

$$R(a|x) = 1- P(\omega_j |\mathbf{x} )$$
$$d^*(x) = argmax_{a \in A} P(a|\mathbf{x})$$

If we build model around $P(a|\mathbf{x})$ directly, this is a \textbf{Discriminative Model}. If We try to model the joint distribution $P(\mathbf{x},a)$, this is a **Generative Model**. Same as we get the Bayesian estimator, we try to find

###### The Independence Assumption

Naive Bayes made the important **assumption of attribute conditional independence** to write

$$\frac{P(a)P(\mathbf{x}|c)}{P(\mathbf{x})}= \frac{P(a)}{P(\mathbf{x})}\prod_{i=1}^n P(x_i|a)$$

We just need to count dataset to get

$$\hat{P}(x_i|a) = \frac{|D_{a,x_i}|}{|D|}$$
$$\hat{P}(a) = \frac{|D_{a}|}{|D|}$$

For continuous data, we can use probability density function to get the estimates. 

###### Lapalcian Correction

In most cases, the smoothing **Laplacian Correction** is needed:
$$\hat{P}(x_i|a) = \frac{|D_{a,x_i}+1|}{|D|+n}$$
$$\hat{P}(a) = \frac{|D_{a}|+1}{|D|+n}$$
$$P(a|\mathbf{x}) = \frac{P(a)P(\mathbf{x}|c)}{P(\mathbf{x})}$$
$$Dir(\mathbf{\alpha}) = \frac{\Gamma(\sum\alpha_i)}{\prod_{i=1}^K \gamma(\alpha_i)} \prod_{i=1}^K x_i^{\alpha_i-1}, \sum_{i=1}^K x_i =1$$

It can be proven, When we using the conjugate distribution of **multinomial distribution** to be the prior distribution and correct the parameter for Dirichlet Distribution to be $N_i+\alpha$ is equivalent for the Laplace Correction.

##### Semi-naive Bayesian Classifiers

Assume certain dependencies between attributes. The most common case is "One-Dependent Estimator". Such as

- Super-Parent ODE
- Tree Augmented Naive Bayes
  - Use the Maximum Weighted Spanning Tree. Weighted by mutual information (conditional entropy), Build a complete graph on attributes.
- Average One-Dependent Estimator
  - Ensemble on the SPODE Models

##### Pros and Cons of Naive Bayes

- Assume independence of features
- Bayesian Method
- Lazy evaluation
- Robust to Missing Values

---

#### K-nearest Neighbor

- No Training stage
- higher  K is, more robust the model can be

---

#### Linear Discriminative Analysis

[LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) and QDA \(Quadratic Discriminative Analysis\) are generative models

- [Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  - Data Sphering
- Generalized Rayleigh Quotient
