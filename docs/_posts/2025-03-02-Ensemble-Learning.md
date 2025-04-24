# Ensemble Learning

xgboost

Cons

1. Iterate dataset multiple times during training
  1. I-O time very high for large dataset if use batch training
  2. Only work on small data if all data is read in memory 


Python语言中LightGBM提供了两种调用方式，分为为原生的API和Scikit-learn API，两种方式都可以完成训练和验证。当然原生的API更加灵活，看个人习惯来进行选择。



Light gbm 

其缺点，或者说不足之处：

每轮迭代时，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。

预排序方法（pre-sorted）：首先，空间消耗大。这样的算法需要保存数据的特征值，还保存了特征排序的结果（例如排序后的索引，为了后续快速的计算分割点），这里需要消耗训练数据两倍的内存。其次时间上也有较大的开销，在遍历每一个分割点的时候，都需要进行分裂增益的计算，消耗的代价大。
对cache优化不友好。在预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化。同时，在每一层长树的时候，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样，也会造成较大的cache miss。
lightGBM特点
以上与其说是xgboost的不足，倒不如说是lightGBM作者们构建新算法时着重瞄准的点。解决了什么问题，那么原来模型没解决就成了原模型的缺点。
概括来说，lightGBM主要有以下特点：
- 基于Histogram的决策树算法
- 带深度限制的Leaf-wise的叶子生长策略
- 直方图做差加速
- 直接支持类别特征(Categorical Feature)
- Cache命中率优化
- 基于直方图的稀疏特征优化
- 多线程优化
前2个特点使我们尤为关注的。