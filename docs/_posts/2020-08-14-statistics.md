---
layout: page
title: "Statistics"
date: 2020-08-14 15:00:00 -0000
categories: Math
---

- [Basic Concepts](#basic-concepts)
  - [Descriptive Statistics](#descriptive-statistics)
  - [Statistical Inference](#statistical-inference)
    - [Theorems](#theorems)
    - [Estimation Methods](#estimation-methods)
      - [Method of Moments](#method-of-moments)
    - [Maximum Likelihood Estimation(MLE)](#maximum-likelihood-estimationmle)
      - [Properties](#properties)
      - [$\Delta$ Method](#delta-method)
  - [Hypothesis Testing](#hypothesis-testing)
    - [Single Variable Distribution Based Test](#single-variable-distribution-based-test)
    - [Test Multiple Variables](#test-multiple-variables)
    - [Computation-based hypothesis Testing Approach](#computation-based-hypothesis-testing-approach)
    - [Test Multiple Hypothesis](#test-multiple-hypothesis)
    - [Computational Approach for Hypothesis Tests](#computational-approach-for-hypothesis-tests)
- [Model Selection](#model-selection)
- [A/B Testing](#ab-testing)

## Basic Concepts

### Descriptive Statistics

- Univariate Statistics
  - shape: variance, [skewness](https://en.wikipedia.org/wiki/Skewness), [kurtosis](https://en.wikipedia.org/wiki/Kurtosis)
  - center: mean, median, mode
  - spread: standard deviation, [entropy](https://en.wikipedia.org/wiki/Entropy_%28information_theory%29)
  - relative position
- Bivariate Statistics
  - correlation, covariance
    - shrinkage
    - [spearman correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
    - [chi-square statistics](https://en.wikipedia.org/wiki/Chi-squared_test)
    - limits of correlation/covariance
      - measures linear relationship
      - dependent variables not correlated
  - regression methods
- multivariate

### Statistical Inference

- parameters
  - constant for probability model)
- statistic
  - model of sample data
- estimator
- data, sample, population  
- point estimation, interval estimation
  - Confidence Interval
    - $P(L \leq \theta \leq U )$ q is not random, L, U is random! 
      - ( We repeat constructing condence inter val a n times, a percent of the times, it will contain t h e t a

#### Theorems

- Independence and Correlation
  - dependent but zero correlation
    - X, X^2 \(normal, Chi-square\)
- Law of Large Number
- Central Limit Theorem
  - i.i.d. assumption
  - regularization condition
  - convergence rate

#### Estimation Methods

Criteria for estimators

- Unbiased $E(\hat{\theta}) = \theta$
- Minimum Variance (MVUE, minimum variance unbiased estimator) $Var(\hat{\theta}) < Var(\theta')$
- Efficient
- Coherent

##### Method of Moments

Estimate $E(X^k)$ based on Law Of Large Numbers

If We have p parameters, we can use p moments to form a system of equations to solve $\theta_1,...\theta_p$
$$\sum_{i=1}^n X_i^j= E(X^j)$$
,for j = 1,...,p

Properties

- Almost surely exist
- Consistent
- Asymptotically Normal (variance decrease at $\frac{1}{\sqrt{n}}$)

#### [Maximum Likelihood Estimation(MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

Multiply p.m.f/p.d.f since every sample is independent. Maximize the likelihood of finding samples. \\ If $X_1,...X_n \stackrel{i.i.d}{\sim} f_x(x, \theta)$, 

$$ l(\theta) =  \prod_{i=1}^n f_{X_i} (x_i; \theta), L(\theta ) = log l(\theta)$$
$$ \hat{\theta}_{MLE}= argmax_{\theta} f_x(x;\theta ) = argmax_{\theta} L(\theta )$$

Analytical or Numerically solved. 

$$\frac{\partial}{\partial \theta} [log L(\theta ) ] = 0, \frac{\partial^2}{\partial \theta^2} [log L(\theta ) ] < 0$$

,for multiple parameters, we need the Hessian matrix to be negative definite $x^tHx<0, \forall x$ 

##### Properties

Invariance
   -  $\hat{\theta}$ is MLE of $\theta$, then $g(\hat{\theta})$ is MLE of $g(\theta)$

Consistency

- $$P(\hat{\theta} - \theta ) \rightarrow 0$$
- as $n \rightarrow 0, \forall \epsilon > 0$ Under the conditions
  1. $X_1,...X_n \stackrel{i.i.d}{\sim} f_x(x|\theta)$
  2. parameters are identifiable, $\theta \neq \theta', f_x(x|\theta) \neq f_x(x|\theta')$
  3. densities $f_x(x|\theta)$ has common support(set of x with positive density/probability), $f_x(x|\theta)$ is differentiable at $\theta$
  4. parameter space $\Omega$ contains open set $\omega$ where true $\theta_0$ is an interior point

Asymptotic Normality 

$$ \sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \rightarrow N(0, I^{-1}(\theta_0)) $$
$$ I(\theta_0) = E( -(\frac{\partial}{\partial \theta} [log f(x, \theta ) ])^2)=E(-\frac{\partial^2}{\partial \theta^2} [log f(x, \theta ) ] )$$

is called the [Fisher Information](https://en.wikipedia.org/wiki/Fisher_information)

$$ \hat{\theta}_{MLE} \approx N(\theta_0, \frac{1}{n I(\theta_0)})$$
$$ n I(\theta_0) = E( -\frac{\partial^2}{\partial \theta^2} log L(\theta) )$$

So the Variance of MLE( $1/ E( -\frac{\partial^2}{\partial \theta^2} log L(\theta) )$) is the reciprocal of amount of curvature at MLE. 

Usually, We can just use the \textit{observed Fisher Information} (curvature near $\hat{\theta_{MLE}}$) instead. ($I(\hat{\theta_{MLE}})$) 
$\frac{1}{ n I(\theta_0)}$ is called **Cramer-Rao Lower Bound.** 

Under Multi-dimensional Case,
 		
$$ I(\theta_0)_{ij} = =E(-\frac{\partial^2}{\partial \theta_i \partial \theta_j} [log f(x, \theta )])$$

$$Hessian \approx nI(\theta_0) Hessian^{-1} \approx nI(\theta_0)$$

when we use numerical approach. 

Under the above **4 conditions**, we need to following to have asympotic normality
	
-  $\forall x \in \chi$, $f_x(x|\theta)$ is three times differentiable with respect to $\theta$, and third derivative is continuous at $\theta$, and $\int f_x(x|\theta) dx$ can be differentiated three times under integral sign
-  $\forall \theta \in \Omega, \exists c, M(x)$ (both depends on $\theta_0$) such that
$$ \frac{\partial^3}{\partial \theta^3} [log f(x, \theta ) ] \leq M(x), \forall x \in \chi, \theta_0-c<\theta < \theta_0+c, E_{\theta_0} [M(x)] < \infty$$

##### $\Delta$ Method

$g(\hat{\theta}_{MLE})$ is approximately

$$N(g(\theta), (g'(\theta))^2 \frac{1}{nI(\theta)})$$

if asymptotic normality is satisfied.

In Multivariate Case:

$$\hat{\theta} \sim N(\theta, \Sigma/n), \theta,\hat{\theta} \in R^p$$
$$g: R^p \rightarrow R^m$$
$$g(\hat{\theta}) \sim N(g(\theta), G\Sigma G^T/n)$$

$$ G = \begin{pmatrix} 
    \frac{\partial{g_1(\theta)}}{\partial{\theta_1}}& \cdots & \frac{\partial{g_1(\theta)}}{\partial{\theta_p}}\\ 
    \vdots & \ddots & \vdots \\
    \frac{\partial{g_m(\theta)}}{\partial{\theta_1}}& \cdots & \frac{\partial{g_m(\theta)}}{\partial{\theta_p}} 
\end{pmatrix} $$

### [Hypothesis Testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)

Basic Logic: a conditional statement is equivalent to its contrapositive statement
  $$ A \rightarrow \neg (\cup B_i) \Leftrightarrow \cup B_i \rightarrow \neg A$$ 
  notice, 
  $$\neg (\cup B_i) = \cap (\neg B_i)$$

- Type I error, (wrongly reject, false reject,$1-\alpha$)
  - Significance $\alpha$
  - Connection with Precision
- Type II error, power (wrongly accept, failed to reject)
  - Power $1-\beta$( Pr( Reject $H_0$ | $H_1$ is True ))
  - Connection with Recall

#### Single Variable Distribution Based Test

[Wald Test](https://en.wikipedia.org/wiki/Wald_test) 
$$T = \frac{\hat{\theta} - \theta_0}{Se(\hat{\theta})}$$
$$ \hat{\theta}_{MLE} \approx N(\theta_0, \frac{1}{n I(\theta_0)})$$
$$T = \frac{\hat{\theta} - \theta_0}{\sqrt{\frac{1}{nI(\theta_0)})}}$$

[Likelihood Ratio Test](https://en.wikipedia.org/wiki/Likelihood-ratio_test)

[Score Test](https://en.wikipedia.org/wiki/Score_test)

- Based on Lagrange Multipliers

Rank based tests

used to test mean, mean-like statistics, not as efficient as Computational based test

- Wilcoxon signed-rank test
- subitem Mann-Whitney U test

#### Test Multiple Variables

Pearson's Chi-Square Test for Independence

$$ U = \frac{X_{ij} - E_{ij}}{E_{ij}}$$
$$ E_{ij} = \frac{X_iX_j}{n}$$

Test Discrete Random Variable vs. Continuous Random Variable

- Test $sup|\hat{F}_1 - \hat{F}_2|$, Y is independent of Z if CDF is the same
- Do with regression (with categorical parameter) - test $\beta$

[Wald Test](https://en.wikipedia.org/wiki/Wald_test) for multivariable distributions

#### Computation-based hypothesis Testing Approach

Permutation tests: 

  - Test $X_1,..X_n \sim F, Y_1,...Y_n \sim G, if F=G$.
  - Use $T = Mean(X_i )- Mean(Y_i)$, each time scramble X and V labels and should not not change the distributions of vectors $X_1,...X_n, Y_1...,Y_n$

Bootstrapping:

- $X_1,...X_n \sim F$ with $T = T(X_1,...,X_n)$, to get the distribution of T, \textbf{sample with replacement.} 
- The belief is $(\hat{\theta} - \theta)$ should behave the same as $(\theta* - \hat{theta})$.
-  The first quantity can be treated like a pivot.  (use $(\theta*_1 - \hat{\theta}_1),...(\theta*_n - \hat{\theta}_n)$ to test.

#### Test Multiple Hypothesis

how to test multiple hypothesis\(FWER vs FDR\)

- Family-wise Error Rate(FWER) the probability of rejecting at least one of at least one null hypothesis 
Under independence, the probability of making mistake when all null are true: P( any type I mistake) = 1-P(no type I mistake for all) = $1-(1-\alpha)^M=\beta$)
- Bonferroni correction, assuming independence 
  - $$P(\bigcup_{i=1}^n \text{typeI mistake}) \leq \sum_{i=1}^n P(\text{typeI mistake}) \leq M\alpha$$
  - control at $\alpha=\frac{\alpha}{M}$
  - $\alpha$ being to small will impact power of the individual tests!
- False Discovery Rate(FDR): 
  - bound the fraction of type-I errors. R be the total number of hypotheses rejected. V be the number of rejected hypotheses that were actually null. Let FDR = V/max(R,1), control $E(FDR) \leq \alpha$.


#### Computational Approach for Hypothesis Tests

- Permutation Test
- BootStrapping
  - what is the percentage left out of sample? \(N -&gt; inf for sampling times and sample size\)

## [Model Selection](https://en.wikipedia.org/wiki/Model_selection)

- ##### [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance#:~:text=Analysis%20of%20variance%20(ANOVA)%20is,by%20the%20statistician%20Ronald%20Fisher.)
  - what if p=thredhold\(0.05\)
    - it's up to you
- [Bias-Variance Decomposition](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [AIC - Akaike Information Criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion)
  - K-L Distance
	- $$D_{KL}(f,\hat{f})) = \int_{-\infty}^{\infty} log(\frac{f_X(x)}{\hat{f(x)}})f_X(x)dx $$
	- $$ = const + \frac{1}{2} \int (-2log\hat{f}(x))f(x)dx = const + AIC $$
	- $$ A(f,\hat{f}) = -2 logL(\theta) + 2 p (\frac{n}{n-p+1})$$
- [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion)

## [A/B Testing](https://en.wikipedia.org/wiki/A/B_testing)

Define metrics

- Direct metrics vs Compound Metrics
- High-level (Unified) Metrics vs Low-level metrics: level of aggregation
- Primary Metric vs Secondary Metrics \\
	(primary metrics - usually explains metric change)

Compare metrics

Experiment Design

- Experiment meaning you have to modify the user/object
- Experiment group vs control group/ treatment group
- Confounding Variables and control of Confounding Variables
- Randomized experiment
  - approximation for treatment effect
  - random sampling -hashing cookie id
- Other sampling methods
	- Stratified Sampling
	- Cluster Sampling
	- randomization - flipping a coin 
- [novelty effect](https://en.wikipedia.org/wiki/Novelty_effect) - the difference between groups are too large
- Techniques
  - Matching
  	- one-to-one match on selected segmentations
  	- set-to-set matching with frequency of selected factors
  	- propensity score matching(propensity score is estimated any classification model, typically logistic regression)
  - Segmentation ： 
    - divide objects to sub-groups
  		- deal with sub-group effect
  		- if you suspect the randomization is not perfect
  - Advanced control group settings
  	- Holdout group (group that you don't expose to new environment) to understand the effect of additional environment changes
  	- Double control group (AAB test, AA test) to detect potential bias of the testing
  - Sample Size (7\% on one day or 1\% over 7 days)
  	- correlations between days (autocorrelation)
  	- day effects
  	- allow to measure long term effects

Hypothesis Testing based on Experiment

- Usually need to approximate discontinuous data using continuous distribution hypothesis test( t-test, Z-test). Convert multi-class metrics to binary metrics (ask yes or no questions)