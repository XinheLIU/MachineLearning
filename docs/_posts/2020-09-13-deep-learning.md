---
layout: page
title: "Deep Learning"
date: 2020-09-13 15:00:00 -0000
categories: Deep Learning
---

- [Artificial Neural Networks](#artificial-neural-networks)
  - [feedforward neural network](#feedforward-neural-network)
    - [multi-layer perceptron](#multi-layer-perceptron)
  - [Radial basis function (RBF) neural network](#radial-basis-function-rbf-neural-network)
  - [Restricted Boltzman Machine(RBM)](#restricted-boltzman-machinerbm)
  - [Self-Organizing Feature Map\(SOM\)](#self-organizing-feature-mapsom)
  - [Fuzzy neural network](#fuzzy-neural-network)
- [Deep Learning Models](#deep-learning-models)
  - [Convolutional Neural Network](#convolutional-neural-network)
    - [Convolution](#convolution)
    - [Pooling](#pooling)
    - [Basic Architecture of Image Classification](#basic-architecture-of-image-classification)
    - [Modern CNN](#modern-cnn)
      - [Inception V3\(2015\)](#inception-v32015)
    - [ResNet \(2015\)](#resnet-2015)
  - [Recurrent Neural Network \(RNN\)](#recurrent-neural-network-rnn)
    - [Gated Recurrent Unit](#gated-recurrent-unit)
    - [Long Short-term Memory](#long-short-term-memory)
    - [Recursive Neural Network](#recursive-neural-network)
  - [Autoencoder](#autoencoder)
  - [Generative Adversarial Networks](#generative-adversarial-networks)
  - [Deep Belief Network](#deep-belief-network)
- [Training Deep Learning Models](#training-deep-learning-models)
  - [Gradient Descent](#gradient-descent)
  - [Deep Neural Network Training](#deep-neural-network-training)
    - [Gradient Vanishing and Explosiong](#gradient-vanishing-and-explosiong)
    - [weight initialization](#weight-initialization)
    - [activation functions](#activation-functions)
    - [Batch Normalization](#batch-normalization)
    - [Regularization](#regularization)
    - [how to debug](#how-to-debug)
    - [\(unsupervised\) training on each layer](#unsupervised-training-on-each-layer)
- [Natural Language Processing Basics](#natural-language-processing-basics)
  - [Basic Concepts](#basic-concepts)
  - [Language Models](#language-models)
    - [Sequence-to-Sequence Models](#sequence-to-sequence-models)
    - [Attention Models](#attention-models)
- [Computer Vision Basics](#computer-vision-basics)
  - [Computer Vision and YOLO Algorithm](#computer-vision-and-yolo-algorithm)
    - [Object Detection](#object-detection)
    - [Landmark Detection](#landmark-detection)
    - [Convolutional Implementation of Sliding Windows + YOLO - You Only Look Once Algo( Bounding Box Detection)](#convolutional-implementation-of-sliding-windows--yolo---you-only-look-once-algo-bounding-box-detection)
    - [Region Proposal](#region-proposal)
  - [Face Recognition](#face-recognition)
  - [Neural Style Transfer](#neural-style-transfer)

## Artificial Neural Networks

### [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network)

#### multi-layer perceptron

$$y = \phi(\sum_{i=1}^N w_ix_i)$$

- Can only solve linear-separable problem
- Its it proven that a single-hidden layer multilayer perceptron can approximate any continuous functions at arbitrary error level.(universal approximation
- [backward propagation](https://en.wikipedia.org/wiki/Backpropagation)
- [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus)
  - [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)

### [Radial basis function (RBF) neural network](https://en.wikipedia.org/wiki/Radial_basis_function_network#:~:text=In%20the%20field%20of%20mathematical,the%20inputs%20and%20neuron%20parameters.)

$$ \rho(\mathbf{x,w_i},\sigma) = exp(\frac{ -{\| \mathbf{x} - \mathbf{w_i} \|}^2}{2\sigma^2})$$

As long as a feature's distance to the center vector ( here $\mathbf{w_i}$) the same, the function value is the same. $\mathbf{w_i}$ separates different hidden unit band with bandwidth $\sigma$ 

The Gaussian function 

$$exp(-\|\mathbf{x} - \mathbf{u_i}\|^2)$$

(like Kernel) can help to transform linear inseparable case as-if projecting to a high-dimension space (same as SVM), to a linear separable case.

Alternatively, treat an RBF as a interpolation solution. It tries to data hyperplane. It reduces the noise by interpolation among the data points. The interpolated hyperplane still passes all data points. 

Compare with Neural Network: both can achieve universal approximation, while RBF network uses a local approximation approach. 

training

1. Initialization of $\mathbf{w_i}$ by random initialization or \textbf{unsupervised learning} like K-means. \\ Usually, we have $\sigma = d_max/\sqrt{2K}$, $d_max$ is the maximum distance between centers. (make sure bandwidth is not too small or too big)
2. Training $\mathbf{w_i}$. Use [recursive least square](https://en.wikipedia.org/wiki/Recursive_least_squares_filter)
	- $$\mathbf{R}(n)\mathbf{\hat{w}}(n) = \mathbf{r}(n)$$
- $\mathbf{R}(n)$ is the covariance matrix between hidden layer outputs ($\hat{y}$), $\mathbf{r}(n)$ is the covariance vector between hidden layer outputs ($\hat{y}$) and model response.
	- Training by solving $\mathbf{R}^{-1}(n)$
3. After training, use Back propagation to train all parameters one more time. (train the whole network after training layers)

### [Restricted Boltzman Machine(RBM)](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)

### Self-Organizing Feature Map\([SOM](https://en.wikipedia.org/wiki/Self-organizing_map)\)

- competitive learning
  - inner product
- typological neighbor
  - single peak
  - transform invariant
  - time variant
- weight initialization
  - PCA Initialization
  
### [Fuzzy neural network](https://en.wikipedia.org/wiki/Neuro-fuzzy)

- fuzzy logic
  - fuzzy set, fuzzy number
- training - no gradient
  - level-set based
  - conjugate gradient
    - genetic algorithm based

## Deep Learning Models

### [Convolutional Neural Network](hhttps://en.wikipedia.org/wiki/Convolutional_neural_network)

#### Convolution

- Convolute the Image Data with a Filter(Kernel) (eg. Sobel Filter, Scharr Filter)
	- $$(n.n) * (f, f) \rightarrow (n-f+1, n-f+1)$$
- Padding: add zeros entries so the output size same as input size
$(n+2p-f+1, n+2p-f+1)$ as result
   - Valid Padding: No Padding out
   - Same Padding: Output the Same Size
   - FULL Padding: Maximum Padding does not result in a convolution on just padded elements(eg. for a filter of size k, k-1)
- Stride:steps to take
   - $(\lfloor \frac{n+2p−f}{s}⌋+1 \rfloor, \lfloor \frac{n+2p−f}{s}\rfloor + 1)$ rounding down (dont do when it is out)
- Convolution over volume: Traditionally, use a filter with same number of channels (each 3-dimensional filter result an matrix output)
- Convolution of tensor - still all number multiply then sum together
could use m filters to result an m channel output
- 1-layer Convolution Network: Input $\rightarrow$ n filters $\rightarrow$ ReLu on each output (bias parameter added here) $\rightarrow$ Stack the output together result an n channel output 
- Why convolution works
   - Parameter Sharing: One feature detector is useful for one part of image probably be useful for all parts(fewer parameters)
   - Sparsity of Connections: In each layer, each output value depends only on a small number of inputs
   - translation invariance/translational equivalence
  - feature map

#### Pooling

  - downsampling layer
  - max pooling
  - how to back-propagate on pooling layer

#### Basic Architecture of Image Classification

Image of Different Channels (RGB)  $\rightarrow$  Conv Layer  $\rightarrow$  Pooling (multiple layers of both)  $\rightarrow$  Fully connect layer (flatten all data)  $\rightarrow$  Sotfmax  $\rightarrow$ Prediction. Use Back propagation to train (Mini-batch gradient descent)

#### Modern CNN

- LeNet \(LeNet-5\)
  - avg pool
	- shrink each step
	- no padding
	- conv-pool, conv-pool pattern
	- activation:sigmoid, ReLu
- AlexNet\(2012\)
  - much bigger network
	- ReLU
	- Multiple GPU Training
	- Local Response Normalization (normalize over all channels)
	- [Grouped Convolutions](https://towardsdatascience.com/grouped-convolutions-convolutions-in-parallel-3b8cc847e851#:~:text=This%20process%20of%20using%20different,convolutions%20on%20a%20single%20image.)
- VGG\(2015\)
  - 2 3x3 convolution vs. 5x5 convolution: add non-linearity, more discriminative
- VGG-16, VGG-19
  - even bigger

##### Inception V3\(2015\)

- inception block
- 1x1 convolutions
- Create a "bottleneck layer"
  - reduce the number of channels and not hurting the quality of the model
- model Gaussian blur filter
  - filter decomposition
  - nxn replaced with 1xn and nx1
- Different (may be one or two layers of) Pooling/Convolutions in one layer, Channel Concatenation
- Identity Blocks connected together

#### ResNet \(2015\)

- Train significant Deeper Networks
- Build by Residual Blocks
	- identity block
	- convolution block
- **Skip Connections/ Short Cut**
$$ a^{[l+2]}= g(z^{[l+2]}) + a^{[l]})$$
- helps gradient propagation
- less likely to learn identity functions
  - residual connections
- Network in Network and 1x1 Convolution
	- Adds no linearity to the neural Network
	- Shrink the number of Channels

### [Recurrent Neural Network \(RNN\)](https://en.wikipedia.org/wiki/Recurrent_neural_network)

RNN is a sequence model. Sequence models deal with sequence data ( speech, music, DNA, Sentiment Classification, etc). In sequence data, Inputs, outputs can be different in size and Input might not share features across sequence (eg. text)

- Use the output from t-n(eg. t- 1) as an input for t prediction 
- the output at each time step is passed by a random sampling with predicted probability
- forward propagation
  - w parameters shared across t (weight sharing)
  - $$a^{<t>} = g(w_{ax} x_t + w_{aa} a^{<t-1>})$$
  - $$y^{<t>} = g(w_{ya} a^{<t>} + b_y)$$
- Vectorization
  - $$[w_{aa}| w_{ax}] = w_a$$
  - $[a_{t-1}, x_t]$ stacked vertically
- Backward propagation
  - $$L^{<t>} (\hat{y_t},y_t) = - y^t log \hat{y_t} - (1-y_t) log(1-\hat{y}_t)$$
  - $$L(\hat{y},y) = \sum L(t) L^{<t>}, \forall t$$
  - weights propagates back from t to ....0

Different Architecture Types

- Standard RNN(many-to-many RNN)
  - encoder + decoder(x RNN connects to y RNN in sequence
- many-to-one RNN (only one output at t)
- One-to-One (only one time piece)
- One-to-Many (only one input at time 0)
- Vanishing Gradient Problem
  - basic RNN has many local influences (influence not far)
  - Exploding: solved by **gradient clipping** - rescale gradient when gradient hits some thredhold
  - Vanishing is harder to solve
- Bidirectional RNN
  - Two activations connected in two different directions
  - ![Bidirectional RNN](/assets/img/Bidirectional.png)
- Deep RNN
  - Stacking RNN Layers together. Three layers are already pretty deep.
  - ![Deep RNN](/assets/img/DeepRNN.png)

#### [Gated Recurrent Unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit)

GRU and LSTM can solve the problem of gradient vanishing and creates long-distance dependencies by "peephole connection". Stacking LTSM or GRU units together creates LTSM/GRU network.

![GRU and LSTM](/assets/img/GRU-LSTM.png)

$$\tilde{c}^{<t>} = tanh( W_c [ \Gamma_r * c^{<t-1>},x^{<t>}] + b_c)$$
$$ \Gamma_u = \sigma(W_u[c^{<t-1>},x^{<t>}] + b_u)$$
$$ \Gamma_r = \sigma(W_r[c^{<t-1>},x^{<t>}] + b_r)$$
$$c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + (1 - \Gamma_u) * c^{<t-1>}$$
$$a^{<t>} = c^{<t>}$$
  
#### [Long Short-term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory)

$$\tilde{c}^{<t>} = tanh( W_c [ c^{<t-1>},x^{<t>}] + b_c)$$
$$ \Gamma_u = \sigma(W_u[c^{<t-1>},x^{<t>}] + b_u)$$
$$ \Gamma_f = \sigma(W_f[c^{<t-1>},x^{<t>}] + b_r)$$
$$\Gamma_o = \sigma(W_o[c^{<t-1>},x^{<t>}] + b_o)$$
$$c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + \Gamma_f * c^{<t-1>}$$
$$a^{<t>} = \Gamma_o c^{<t>}$$

- memory cell, input gate, output gate, forget gate
- outter product
- peehole connection
  - solves vanishing gradient

#### [Recursive Neural Network](https://en.wikipedia.org/wiki/Recursive_neural_network) 

- is a more general form of Recurrent Neural Network

### [Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)

- sparse autoencoder
  - L1 on activations, sparse code
- denoising autoencoder
  - add noise
    - Generative noise
       - $$P(\tilde{X}|X)$$
    - $$max P(X|(\phi \circ\psi)\tilde{X})$$
  - dropout
- contractive autoencoder
  - loss function with regularization
- variational autoencoder
  - Bayesian Model
  - hidden layer parameter ditributed normal
- training
  - unsupervised training on layer
  - supervised fine-tune
  - learn to fix distortion - add distortions in the input

### Generative Adversarial Networks

- Generative model
  - L = \|\|f\(img\) - f\(Gen\(img\)\)\|\|, f is a neural net
    - VGG
- Generator + Discriminator
  - Generator takes noise, paramers
  - Discriminator takes fake and real images
- advantages
  - model free
    - can learn different priors / loss functions
- disadvantage
  - lack of theoretical foundation
  - hard to train
    - reach Nash equilibrium
- application
  - style transfer

### [Deep Belief Network](https://en.wikipedia.org/wiki/Deep_belief_network#:~:text=In%20machine%20learning%2C%20a%20deep,between%20units%20within%20each%20layer.)

- [restricted Boltzman Machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
  - training
    - contrastive divergence
    - positive gradient
- [Deep belief network](https://en.wikipedia.org/wiki/Deep_belief_network)
  - training by layers + fine tune

## Training Deep Learning Models

### [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)

$$w_i(t+1) = w_i(t) + \eta [d_j-y_j(t)]x_{j,i}$$

[Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- Noise Reduction
  - dynamic sampling
  - iterative averaging
  - gradient averating
  
Mini-batch gradient descent

- Use one batch (subset) of sample to compute the gradient each time. ( one “epoch”) ( one batch size =1, it is Stochastic gradient descent)
  
  - Momentum on Gradient
    - $$w_t = w_{t-1} - h_t, h_t = \alpha h_{t-1} + \eta_t g_t$$



Momentum Methods

- Smooth the gradient series with EWMA (Exponentially weighted averages)
	$$ V_{dw} = \beta_1V_{dw} + (1-\beta_1) dw, V_{dw} /= (1-\beta_1^t)$$
 	$$ V_{db} = \beta_1V_{db} + (1-\beta_1) dw,V_{db} /= (1-\beta_1^t)$$
- Nesterov Momentum
- AdaGrad
  - Adaptive Methods to eliminate learning rates sensitivity
- Root-Mean Square Prop (RMSProp)
 	$$ S_{dw} = \beta S_{dw} + (1-\beta) dw^2 $$
 	$$ S_{db} = \beta S_{db} + (1-\beta) dw^2 $$
	$$ w:= w- \alpha \frac{dw}{\sqrt{sdw}}, b:= b- \alpha \frac{db}{\sqrt{sdb}}$$
- Adam(Adaptive Moment Estimation) Algorithm that Combines RMSProp and Momentum
	$$ V_{dw} = \beta_1V_{dw} + (1-\beta_1) dw, V_{dw} /= (1-\beta_1^t)$$
 	$$ V_{db} = \beta_1V_{db} + (1-\beta_1) dw,V_{db} /= (1-\beta_1^t)$$
 	$$ S_{dw} = \beta_2S_{dw} + (1-\beta_2) dw^2 $$
 	$$ S_{db} = \beta_2S_{db} + (1-\beta_2) dw^2 $$
	$$w:= w- \alpha \frac{v_{dw}}{\sqrt{sdw}+\epsilon}, b:= b- \alpha \frac{v_{db}}{\sqrt{sdb}+\epsilon}$$
	$$w:= w- \alpha \frac{dw}{\sqrt{sdb}+\epsilon}, b:= b- \alpha \frac{db}{\sqrt{sdb}+\epsilon}$$
- Learning Rate Decay
  $$ \alpha = \frac{1}{1+\text{decay rate} \times \text{epoch num}}$$	


### Deep Neural Network Training

Characteristics of Deep-Learning

- advantage of Deep-learning is significant mostly with large data set.
- Traditional bias-variance trade-off can largely be overcome by adding more data (reducing variance) and training a larger network(reducing bias) cycle when data is sufficient.
- Optimization becomes more crucial in the training process. Dataset normalization, gradient checking are needed. Initialization carefully to avoid 

#### Gradient Vanishing and Explosiong

- [Gradient Vanishing](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- Gradient Explosion

#### weight initialization

- criteria
  - E\(sum\(x_i w_\_i\)\) = 0
  - Var\(sum\(xiwi\)\) = 1
- Types
  - Xavier initialization
  - He Initialization

#### activation functions

- Vanishing Gradient and Exploding Gradient
- sigmoid
- tanh
- ReLU \(Rectified Linear Unit\)
  - ["dying ReLU"](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks) problem
    - sparsity
  - Leaky ReLU
  - exponential ReLU
- evaluate activation functions
  - gradient fast to compute
  - gradient vanishing/exploding
  - faster convergence
  - zero-centered
  - dying neuron \(dying ReLU\)

#### Batch Normalization

The distribution of activation inputs' distributions get closer to upper and lower limits when we go deeper(gradient vanishing), use normalization to normalize the distribution and get larger gradient

- estimate mu and sigma \(exponential smoothing\)
- normalize neuron output before activation
- Regularization
  - dropout
- training a neuron present with probability p testing always present

#### Regularization

- data augmentation
  - Adjust color (use PCA to analyze RGB values, add noises on each direction)
  - Random rotate, crop, shift, change size, symmetric transform
  - Add noise (Gaussian Noise, Salt-and-Pepper Noise)
  - Change resolution, sharpness, contrast
  - Use Generative Model to generate new samples(GAN)
  - Extract features first, then use up-sampling/SMOTE
- dropout
  - works better on large amount of data
- weight sharing
- activation regularization
  - MaxOut
- regularization term
  - weight decay
    - L1, L2 Norm
    - $$J(z) = \frac{1}{m} \sum_{i=1}^m L(y_i,\hat{y}_i) + \sum_{l=1}^L\frac{\lambda}{2m} \|\mathbf{w}^{[l]}\| ^2_F$$
- training process regularization
  - early stopping

#### how to debug

- gradient checking

#### \(unsupervised\) training on each layer

- proposed by Hinton for restricted Boltzman machine
- help on gradient vanishing
- as effective as good weight initialization

## Natural Language Processing Basics

### Basic Concepts

- Text, character, token/word
- Bag of Words
- Word Embedding
  - one-hot encoding
  - Word2Vec
    - CBOW
    - Skip-Gram
    - hierarchical softmax, negative samples
  - GloVe
- [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
- Deep Learning in NLP
  - Mixture Density Network
    - Deep Neural Network and Gaussian Mixture Model
  - Feature Selection
    - Mel Frequency Cepstral Coefficient
    - Hidden Markov Model + Gaussian Mixture Model

### Language Models

Language model uses data from a **Corpus**. It models probabilities of words appear(generative model).

The input is a language sequence, prediction is the (conditional) probability of next word appearing. The prediction is a Sampling (according to predicted probability until we hit EOS).

Application examples like；

Sentiment Classification:

$$\text{word embedding} \rightarrow Average \rightarrow softmax \rightarrow \hat{y}$$

Or a *many to one* RNN with sentiment as the (only) output at the last step.

#### Sequence-to-Sequence Models

![Sequence to Sequence Model](/assets/img/SequenceToSequence.png)

- Use [Beam Search](https://en.wikipedia.org/wiki/Beam_search#:~:text=In%20computer%20science%2C%20beam%20search,that%20reduces%20its%20memory%20requirements.) to train

usually refine to

$$max_y  \frac{1}{T_y^{\alpha}} \sum_{t=1}^{T_y} log P(y^{<t>} | x, y^{<1>},...,y^{<t-1>})$$

to avoid a preference for short sentence (length normalization)

Error Analysis:

By comparing $P(y^* |x)$ and $P(\hat{y}|x)$ we can attribute the error to beam search or RNN(if searched probability lower than correct probability-Beam Search’s fault)

Translation problem measure - Bleu Score (See paper  textit{method for automatic evaluation of machine translation})

$$p_n = \frac{\sum_{ngram \in \hat{y}} count_{clip} (ngram)}{\sum_{ngram \in \hat{y}}count(ngram)}$$

#### Attention Models

To Solve the problem sequence-to-sequence model faces in long sentences (information become less useful when too far). We add double hidden state and parameters to represent the attention paid to models.

![Attention Model](/assets/img/AttentionModel.png)

Attentions are trained by a small-neural network 
$$a_{<t,t^`>} = \frac{exp(e_{<t,t^`>} ) }{\sum_{t^`=1}^{T_x} exp(e_{<t,t^`>}) }$$
$$f(a_{<t’>}, s^{<t-1>}  ) = e_{<t,t^`>}$$

## Computer Vision Basics

### Computer Vision and YOLO Algorithm

#### Object Detection

- Can have different objects (several objects)(little different than classification with detection)
- Output a vector: $[p_c, b_x,b_y,b_h,b_w, c_1,c_2...c_n]$ for both location(center + bounding box) and class
	-  use square loss function
- IOU - Intersection under union
	-  Intersection of two bounding boxes/union of boxes ( predicted and actual) 
	-  thredhold usually 0.5

#### Landmark Detection

Output x y coordinates for important points in the image

#### Convolutional Implementation of Sliding Windows + YOLO - You Only Look Once Algo( Bounding Box Detection)

- Use convolution to replace （two) fully connected layers( two layers ) in the network
	-  pic $\rightarrow$ Deep CNN  $\rightarrow$ encoding of dimension(m, grid, grid, $n_{anchor boxes}$, $n_{features}$)
	-  $n_{features}$ is dimension of $[p_c, b_x,b_y,b_h,b_w, c_1,c_2...c_n]$
- Assign center of the object to the grid
- \textbf{Non-max suppression}: detect the object \textbf{only once}
	-  Use the $p_c$ probability, only keep the one with highest $p_c$ intersected rectangles
	-  discard any box below a $p_c$ thredhold
	-  once for each output class
- Anchor Boxes
	-  different shape boxes to deal with overlapping objects
	-  output becomes $[p_c, b_x,b_y,b_h,b_w, c_1,c_2...c_n]$ for two anchor boxes
	-  each object assigned to the center grid cell and anchor box IOU

#### Region Proposal

- Segmentation Algorithm to find blob and only run on bounding box on these
- R-CNN and Fast R-CNN algorithm, Faster Algorithm

### Face Recognition

- Verification: confirm identity
- recognition: database of K persons, output id if any of the K persons
- One-shot Learning: learn using just one example
- Learn a similarity function: $d(p1,p2) \leq t$， the same person
- \textbf{Siamese network}: use neural network as encoding. $|f(x(i)) - f(x(j))|$ is small
-  Triplet Loss: encoding of the anchor-positive smaller than anchor-positive.
$$L(A,P,N) = max(( |f(A) - f(P)|^2 - |f(A) - f(N) |)^2 + \alpha, 0 )$$
choose triplet that is hard rather than randomly to improve computational efficiency
- $$\hat{y} = \sigma( \sum w_i|f(xi)k - f(xj)k| +b )$$
- usually use pre-computed network to make prediction

### Neural Style Transfer

- $$J(G) = \alpha J_{content}(c,G) + \beta J_{style}(S,G)$$
- Content cost: use pre-trained Conv Net (eg. VGG Net)
  - $$J(C,G) = 1/2|a[l](c) - a[l](G)|^2$$
  - (L2-norm as cost)
- Style Cost: correlation should be the measure of closeness in style. Use \textbf{Style matrix} - i,j,k is on dimension H,W,C
  - $$G_{kk}^l = \sum_i\sum_j a_{ijk}^{[l]} a_{ijk}^{[l]}$$ 
  - the gram matrix
  - $$J(S,G) = |G[l](S) - G[l](G)|_F^2$$ 
  - normalized 
