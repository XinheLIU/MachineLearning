---
layout: post
title: "Supervised Learning: Regression"
author: "Your Name"
categories: Classic
tags: [machine learning, regression, statistics]
image: linear-regression.jpg
---

- [Linear Regression](#linear-regression)
  - [Intuition - based on Least Square Distance](#intuition---based-on-least-square-distance)
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
- [Implementation](#implementation)

## Linear Regression

Linear regression is one of the foundational algorithms in machine learning. It serves as a starting point for understanding more complex algorithms and provides a simple yet powerful way to model relationships between variables.

## What is Linear Regression?

Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data.

### Intuition - based on Least Square Distance

- Geometric Interpretation of Least Squares
  - indeed a projection from **y** to **X** vector space
- Likelihood and MLE

### Assumptions

- Linear Relationship between covariates and dependent variable
- $E(\varepsilon)=0$
- $Var(\varepsilon) =\sigma^2$: Homoscedasticity
- $\varepsilon$ is independent with covariates 
- x is observed without error (and no perfect multicollinearity in multivariate case)
- (optional, Gauss-Markov Theorem) $\varepsilon$ is normal - when it is, OLS and MLE agrees and to be BLUE(Best Linear Unbiased Estimator)

### Linear Regression Model

#### Least Square Approximation

Under Normal Condition, we have
$$y \sim N(\beta_0 + \beta x_i, \sigma^2)$$
$$ L(\theta) = (\frac{1}{\sqrt{2\pi} \sigma})^n exp( - \frac{\sum_{i=1}^n (y_i - (\beta_0 +\beta_1x_i))^2}{2\sigma^2})$$

Equivalent to minimize

$$RSS(\theta) = \sum_{i=1}^n (y_i - (\beta_0 +\beta_1x_i))^2$$
$$\partial_{\beta_i} RSS = 0, i =0,1$$
, we get
$$ r_{xy} = \frac{s_{xy}}{s_xs_y}, \beta_1 = r_{xy}\frac{s_y}{s_x} =\frac{s_{xy}}{s_x^2},\beta_0 = \bar{y}-\hat{\beta}\bar{x} $$

#### Multivariate Case

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

The Diagonal Entries $h_{ii}$ are the **Leverages.**

#### MLE(Maximum Likelihood) Solution

Assuming noise is normal, maximize

$$p(\mathbf{x_1, x_2 ... x_n}| \mathbf{ w} ) = \prod_k \frac{1}{\sqrt{2\pi} \sigma} exp[ -\frac{1}{2\sigma^2} (y_k - \mathbf{w_t x_k} )^2 ]$$ 

Another matrix representation

$$f(\beta) = min (Y-X\beta)^T(Y-X\beta), f'(\beta) = 2X^T(Y-X\hat{\beta}) = 0$$ 

to solve $\hat{\beta}$

$$ min ||y_k - \mathbf{w^T x}_k ||^2 + \lambda ||\mathbf{w}||_1 $$

#### Prediction Confidence

Variance Error In Prediction

$$V(\hat{y^*} - y^*) = \sigma^2 + \sigma^2[\frac{1}{n} + \frac{x^*-\bar{x})^2}{(n-1)s_x^2}]$$
$$ = V(E(y^*) - y^*) + V(\hat{y^*} - E(y^*)) + 2 cov(\hat{y^*} - y^*, \hat{y^*} - y^*)$$

The cross term is zero, the first term is variance with $\varepsilon^*$, second term is variance in $\beta$.

The confidence interval is $\hat{y^*} \pm t_{\alpha/2, n-2} SE(\hat{y^*})$. 

$R^2$, the coefficient of determination: The proportion of the sum of squared response which is accounted by the model relative to the model with no covariance. (take mean of response)

$$R^2 = 1- \frac{\sum(y_i-\hat{y_i})^2}{\sum(y_i-\bar{y})}$$

Note that $0 \leq R^2 \leq 1$ It only tells predictive power if the model is a good fit.

Adjusted $R^2$: $R^2$ + penalty P

### Testing of Assumptions

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

### Resolutions of Assumption Violations

- Linear Relationship
  - non-linear regression, generalized linear models
  - Transformations ( for outliers, heteroskedasticities, etc)
  - Use different models on different periods/data
- Outliers and Heteroskedasticity
  - Weighted Least Squares regression,  (for outliers, heteroskedasticity) 
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

### Model Selection

- Exhaustive Search by AIC or BIC (more stable than LOOCV)
- **Stepwise Regression/Stepwise Variable Selection** (At each step one covariate is added or dropped)
- Cross-Validation 
  - Leave-one-Out cross Validation of Linear Regression: Prediction Error Sum of Squares 
    - $$PRESS = \frac{\sum(y_i-\hat{y}_{-i})^2}{n}$$
    - $$y_i-\hat{y}_{-i} = \frac{\hat{\varepsilon}_i}{1-h_{ii}}$$ 
      - h is the leverage (hat matrix)

#### Regularization, Ridge, Lasso

##### Ridge

Thikhonov Regularization

$$Rss + \lambda \sum_{i=1}^p \beta^2$$

$\lambda$ is the regularization parameter. The result of Ridge is a **Shrinkage** of $\hat{\beta}$ towards zero.

Note

- No penalty for $\beta+0$ or b.
- The predictors should usually be standardized prior to fitting
- Choose $\lambda$ by cross-validation

##### Lasso

Lasso(Least Absolute Shrinkage and Selection Operator)

$$Rss + \lambda \sum_{i=1}^p |\beta|$$

Can be extended to 
$$-log likelihood + \lambda \sum_{i=1}^p |\beta|$$

Extensions

- Group Lasso
  - group predictors together to be either included or excluded. 
- [Elastic Net](https://en.wikipedia.org/wiki/Elastic_net_regularization)
  - $$Rss + \lambda \sum_{i=1}^p (\alpha|\beta_j| + (1-\alpha) \beta_j^2)$$

## Bayesian Linear Regression

Bayesian linear regression provides a probabilistic framework for regression analysis. It allows us to incorporate prior knowledge and obtain uncertainty estimates for our predictions.

[Content to be expanded]

## Generative Linear Models

- Non-linear regression models
  - Nonparametric Regression
    - Complexity controlled by the smoothing parameter (bandwidth).
    - model complexity interpreted in *Degrees of Freedom/Effective degrees of freedom/equivalent degrees of freedom Residual Degrees of freedom- is n minus model degrees of freedom.
  - Local polynomial Regression
    - only fit a **neighborhood** of a target point. parameter $\alpha$ to control the span-traditionally, 0.5. When weighting the data in the neighborhood,
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
        - $\lambda$ penalty for wiggy function. search of x can be a combination of **basis functions** (n + 4 basis functions, n is the knots)
    - Cubic Splines
- Generalized Additive Models

## Implementation

For practical implementation examples using Python and popular machine learning libraries like scikit-learn, check out our [GitHub repository](https://github.com/yourusername/MachineLearning). 