---
layout: page
title: "Bayesian Methods"
date: 2020-08-20 15:00:00 -0000
categories: Bayesian
---
<!---![image](./assets/Bayesian-Methods.png)--->

## Basic Concepts

* [Bayesian Interpretation of Probability](https://en.wikipedia.org/wiki/Bayesian_probability)
  * $$P(\theta|X) = \frac{P(X|\theta) P(\theta)}{P(X)}$$
    * $P(\theta \vert X)$is posterior probability
    * $P(\theta)$ is prior
    * $P(X \vert \theta)$ is likelihood
    * $P(X)$ is evidence
* [Maximize a Posterior](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)\(MAP\) estimation
* Bayesian Estimator
  * $$\hat{\theta}_{Bayes} = E(\theta | X) = \int \theta \pi(\theta |X ) d\theta$$
* [Conjugate Prior](https://en.wikipedia.org/wiki/Conjugate_prior) 
  * [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution#Bayesian_inference)
    * with Bernoulli Distributin
  * [Gamma Prior](https://en.wikipedia.org/wiki/Gamma_distribution#Conjugate_prior)
    * with normal distribution
      * $$p(\gamma)  = \Gamma(a,b), p(\gamma|x) \propto (\gamma^{\frac{1}{2}} e^{-\gamma\frac{(x-\mu)^2}{2}})(\gamma^{a-1}e^{-b\gamma}) = \Gamma(a+ \frac{1}{2}, b+ \frac{(x-\mu)^2}{2})$$
* [Entropy](https://en.wikipedia.org/wiki/Entropy_%28information_theory%29)
* [Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
  * [zero-avoiding and zero-forcing properties](https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/)
    * [Inclusive and Exclusive](https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/)
* [Bayesian Decision Theory](https://www.cc.gatech.edu/~hic/CS7616/pdf/lecture2.pdf)
  * Bayesian Risk

## Models

### Model Type Concepts

* [Probabilistic graphical models](https://en.wikipedia.org/wiki/Graphical_model)\(PGM\)
* [Generative Models](https://en.wikipedia.org/wiki/Generative_model)
* [Latent Variable Models](https://en.wikipedia.org/wiki/Latent_variable_model)

### Probabilisitic Graphical Models

* Naive Bayes
  * [Laplacian Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
    * Bayesian Interpretation of Laplacian Smoothing
* [Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network)
  * d-connected, d-seprated
  * Markov Blanket
  * Independent Map\(I-Map\)
* [Markov Random Field](https://en.wikipedia.org/wiki/Markov_random_field)
  * factor/potential function
  * Gibbs Distribution
    * Boltzmann Distribution
    * [Hammersley-Clifford Theorem](https://en.wikipedia.org/wiki/Hammersley–Clifford_theorem)
  * Markovianity
    * global, local, pairwise
  * clique
  * chordal graph
    * triangulation
* Gaussian Bayesian Network
* Gaussian Markov Random Field
  * information form of Gassuian
    * $$p(s) \propto exp[-\frac{11}{2}\mathbf{x^TJx} + (J\mathbf{\mu})\mathbf{x}]$$
  * edge potential
* Hybrid Network
  * conditional linear model - parent is discrete
  * threshold model - parent is continuous
* Dynamic Bayesian Network
  * [Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
    * properties
      * sequence model
      * generative model
      * mixture model
    * training
    * application
      * Filtering
        * estimate hidden variable with estimations till now
      * Smoothing
        * estimate hidden variable with observations
      * Decoding
        * estimate hidden sequence with maximum posterior probability
  * [Linear Dynamical Systems](https://en.wikipedia.org/wiki/Dynamical_system#Linear_dynamical_systems)
  * [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter)
    * Linear Kalman Filter
      * recursive Bayesian estimation
      * Kalman Gain
    * extended Kalman Filter
    * unscented Kalman Filter
    * [Particle filter/sequential Monte Carlo](https://en.wikipedia.org/wiki/Particle_filter) 
      * Particle filtering uses a set,can take any form required.)
  * Linear Chain Random Field
  * Conditional Random Field
    * Discriminative

### Latent Variable Models

* [Bayesian Regression](https://en.wikipedia.org/wiki/Bayesian_linear_regression)
* [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)
* [Mixture Models](https://en.wikipedia.org/wiki/Mixture_model)
* [Probabilistic PCA](https://www.cs.ubc.ca/~schmidtm/Courses/540-W16/L12.pdf)
* Mean-field models
  * [Ising Model](https://en.wikipedia.org/wiki/Ising_model)
* [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
  * [Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
    * Multinomial Distribution
* Bayesian Neural Network
  * [Langevin Monte Carlo](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) Methods
* Variational Autoencoder

## [Expectation-Maximization Algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)

* E-Step
  * $$q_{k+1} = argmin_q\mathcal{KL}[q(T)||p(T|X, \theta_k)]$$
* M-Step
  * $$\theta_{k+1} = argmax_\theta E_q[log p(X,T|\theta)]$$

* deal with missing data
* sequence of simple tasks instead of optimization
* guaranties convergence
  * only local maximum
* extensions
  * variational E-step
  * sampling M-step

Example

Train a Gaussian Mixture Model

$$argmax_{\theta} E_{q} log p(X, T|\theta)$$

$$log p( X|\theta) = \sum log p(x_i|\theta) = \sum log \sum \frac{q(t_i = c)}{q(t_i=c)}p(x_i, t_i = c |\theta) \geq$$

$$\sum \sum q(t_i=c) log \frac{p(x_i, t_i = c |\theta)}{q(t_i = c)} = \mathcal{L}(\theta, q )$$

* E-Step:
  * $$q_{k+1} = argmax_q \mathcal{L}(\theta^k, q)$$
  * $$log p(X|\theta) = \sum_i \sum_c \frac{q(t_i=c)}{q(t_i=c)} log p(x_i,t_i =c|\theta)\geq \sum_i \sum_c q(t_i=c) log \frac{p(x_i,t_i =c|\theta)}{q(t_i=c)} = \mathcal{L}(\theta,q)$$
  * $$log p(X|\theta) - \mathcal{L}(\theta, q) = \sum \mathcal{KL} (q(t_i) || p(t_i | x_i, \theta))$$
  * $$\mathcal{L}(\theta,q) = \sum_i \sum_c q(t_i=c) log \frac{p(x_i,t_i =c|\theta)}{q(t_i=c)}$$
  * $$=  \sum_i \sum_c q(t_i=c) log p(x_i,t_i =c|\theta) - \sum_i \sum_c q(t_i=c) log q(t_i=c)$$
  * $$ = E_q  log p(X, T|\theta) + const$$

(usually use concave function to optimize)

$\mathcal{L}$ is a variational lower bound here

* M-Step:
  * $$\theta_{k+1} = \argmax_{\theta} \mathcal{L}(\theta, q^{k+1})$$

Notice the usage of Jensen's inequality here.

Train a K-Means Model

* K-Means is actually Gaussian with fixed covariance matrix $\Sigma=I$
* variational q is a set of delta functions
  
## Bayesian Inference

[Inference vs Learning](https://stats.stackexchange.com/questions/205253/what-is-the-difference-between-learning-and-inference)

### Exact Inference

* variable elimination
  * [sum-product variable elimination](https://en.wikipedia.org/wiki/Belief_propagation#Description_of_the_sum-product_algorithm)
* [Belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)
  * clique-tree
  * collect
  * distribute

### [Approximate Inference](https://en.wikipedia.org/wiki/Approximate_inference)

#### Deterministic Approximation

Determinstic Approximation is a type of analytical approximation.

* [Variational Inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
  * [Mean-Field Approximation](https://en.wikipedia.org/wiki/Variational_Bayesian_methods#Mean_field_approximation)
  * [Variational Message Passing](https://en.wikipedia.org/wiki/Variational_message_passing)
  * Variational Bayes
  * Variational EM
  
#### Stochastic Approximation

Stocahstic Approximation is a type of numerical approximation.

* [Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC)
  * Use sample distribution to get posterior distributions statistics
  * [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
  * Monte Carlo Simulation
    * [variance reduction](https://en.wikipedia.org/wiki/Variance_reduction)
  * [Metropolis-Hastings Algorithm](https://en.wikipedia.org/wiki/Gibbs_sampling)
    * detailed balance
  * [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)
    * reduce multi-dimensional sampling to sequential sampling
    * highly correlated samples, slow convergence
  * combine with Metroplis-Hasting to parallelize

[High-Level Explanation of Variational Inference](https://www.cs.jhu.edu/~jason/tutorials/variational.html)

### Bayesian Optimization

* [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process)
  * Kernels
    * Gram Matrix
    * RBF Kernel - squared potential
    * Rational Quadratic
    * White noise
* Bayesian Optimization
  * Surrogate model $\hat{f} \approx f$
    * use information from function
  * sample the next point at
    * upper confidence bound
    * maximize probability of improvement
  * Pros and Cons \(vs Search methods\)
    * hard to parallelize
    * deal better with scarcity/high dimensionality of data
      * when data is expensive