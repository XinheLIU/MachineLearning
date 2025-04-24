  1. Synchronization vs Asyc(https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf).
    1. Dense: sync (CV, NLP)
[Image]
  2. Batch & learning rate
    1. 根据每层权重与权重梯度比值动态adjust学习率
    2. Learning rate decay
  3. Norm
    1. Why needed: Normalization is an important technique in deep learning for several reasons:
      1. Stabilizing Training: Normalization helps stabilize the training process by ensuring that the input features have a similar scale. This can prevent the gradients from becoming too large or too small, which can lead to slow convergence or even cause the training process to fail.
      2. Improved Generalization: Normalization can help the model generalize better to unseen data by reducing the likelihood of overfitting. By bringing all features to a similar scale, normalization can prevent the model from giving undue importance to certain features simply because of their larger magnitude.
      3. Faster Convergence: Normalizing the input features can lead to faster convergence during training, as it helps the optimization algorithm find the optimal solution more quickly.
      4. Better Conditioning: Normalization can improve the conditioning of the optimization problem, making it easier for the optimization algorithm to find a good solution.
      5. Common normalization techniques include z-score normalization (standardization), min-max scaling, and batch normalization. Each technique has its own advantages and is suitable for different scenarios, but the overall goal is to ensure that the input features are on a similar scale, which can significantly improve the performance and stability of deep learning models.
    2. Batch norm : input feature数相同但分布不稳定 （eg 图像大小固定 像素变化大）
      - Application: BatchNorm is typically applied to convolutional neural networks (CNNs) and deep feedforward networks (MLPs) where mini-batches of data are processed together.
      - Normalization Level: BatchNorm normalizes the summed inputs over the mini-batch dimension. It computes the mean and variance across the batch and applies normalization using these statistics.
      - Example: Suppose you have a CNN that processes images in batches of 32. During training, BatchNorm would normalize the input activations for each batch of 32 images by adjusting their mean and variance based on the statistics of that specific batch.
    3. Layer norm：
      1. Batch Normalization (BatchNorm):
      2. Layer Normalization (LayerNorm):
        - Application: LayerNorm is commonly used in recurrent neural networks (RNNs) and transformers, where the input shapes can vary across different time steps or attention heads.
        - Normalization Level: LayerNorm normalizes the summed inputs over the feature dimension. It computes the mean and variance for each individual feature independently and applies normalization using these statistics.
        - Example: Consider an RNN language model where sentences of varying lengths are processed. LayerNorm would normalize the input activations at each time step independently, regardless of the batch size, by adjusting their mean and variance based on the statistics of that specific time step.
        - In summary, BatchNorm operates on the batch dimension and is suitable for scenarios with consistent input shapes within a batch. LayerNorm operates on the feature dimension and is more suitable for scenarios with variable input shapes across different time steps or attention heads. Both techniques help stabilize the training process, reduce internal covariate shift, and improve the overall performance and generalization of deep neural networks.
  4. Initialization
    1. https://proceedings.mlr.press/v9/glorot10a.html
    2. https://towardsdatascience.com/weight-initialization-in-deep-neural-networks-268a306540c0?gi=a442926ce659