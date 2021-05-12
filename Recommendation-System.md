# Recommendation System

极客时间-AI 内参 推荐系统概述 Recommendation System

input
- user-product matrix （input is score)
- objective: fill in missing scores
	- collaborative filering: based on other values
	- content-based method


https://mp.weixin.qq.com/s/pOeTjOJeTdNp8x4e2yB53w


推荐模型


recommendation system

58 
基于流行度
popularity-based recommendation

popularity is biased, cannot use absolute clicks/visit

- time  of the day
- position on webpage

usually, we use ratios to  estimate/measure popularity

- click-through
- purchase probability

to get unbiased estimates, we need to get unbiased data.( we need to use existing items, while show new items to users) ways are

- Exploitation & Exploration
- Epsilon-Greedy

In more advanced approach

- we use Bayesian estimates with priors on these probability estimates, (model the Bernoulli probability for clicks) with priors
- use T-1 estimates to smooth T estimates (Bayesian prior or Temporal Discount)

drawback

- same recommendation for every user, not customized

59

Neighborhood model 基于相似信息的推荐
user-based

- cosine similarity on users
	- user-user matrix


drawback

- rely on enough data for similar user or items(data sparsity)
- user matrix becomes too large
- not interpretable

item-based

- cosine similarity
	- similarity from user feedback to items
	- inverse user frequence: infrequent users contribute more

- user based focus more on user groups
- item based focus more on individual users

drawbacks

- item update slow
- item sparse (comparing to users)
- applies to user fixed scenarios (eg online shopping)

- collaborative filtering
	- use close neighbors as proxy of the users
	- matrix factorization
		- user defined by k features (topics)
		- item defined by k attributes features
		- multiply user * item vector = ranking score

$$L = \sum (r_{ui} - x_u^T y_i) ^ 2 + \lambda_x \sum ||x||^2  \lambda_y \sum ||y||^2 $$


1. define similar users
    1. pearson correlation
        1. weighted Pearson correlation - more weights to user with more data
    2. clusters
    3. entropy/mutual information
2. define an threshold to find user groups
    1. K-groups, K-users, correlation > K
3. use similar items to get estimates
    1. weighted average
    2. baseline (average of all users) + group bias (bias from same group items)

- symmetry
	- we can use similar user to approximate an user or similar items an user viewed to approximate an item





60 
content-based recommendation system



most general form

- use features to represent the interactions of user, items and relationships, define a supervised learning problem

1. feature engineering - represent all information of users and items
    1. item features
        1. text mining of item names, descriptions, 
            1. topic model
            2. word embedding
        2. categorical information
            1. label of items, articles (Item types, article topics)
            2. calssification from other information
            3. inferred from Knowledge Graph 
        3. image and multimedia information
            1. dense vector to a classifier
            2. then combine with other features
        4. user features
            1. demographic : age, gender, location
            2. user group 
            3. user profiling from user feedback/information
- implicit feedback modeling
	- clicks, browsing time, history, 

- no negative feedback -> probably like
- implcit feedback contains noise -> purchase does not mean like
- implcit feedback means confidence, explicit feedback mean preference
	- eg. ranking of a movie vs browsing time on movie reviews
- implcit feedback based RS needs appropriate model evaluation

1. objective function design - very hard
    1. rating and rating related 
        1. fitness rating
    2. click-through, purchase rate
    3. browsing time

Cold Start Problem

user

- rule based: based on registration info (age, gender, location)
- use user information from social network (redirect to SNS)

item

- user based: not a big problem
- item based : periodcally exploit, explore




61 latent variable based

eg.

matrix factorization

- decompse 1e6 x 1e4 user-item rating matrix to two 1e6x100 100x1e4 matrix product
- assume 100 latent varaibles
- dimension reduction

drawbacks

- not explanable
- hard to tune the number/form of latent variable models

62

矩阵分解的流行起源于 10 年前的 Netflix 大赛，当时各类矩阵分解模型都在实际数据中起到了很好的效果。

今天我们要分享的模型，叫作“基于回归的隐变量模型”（Regression-based Latent Factor Model）。这是在基本矩阵分解的基础上衍生出来的一类模型。

Regression-based Latent Factor Model

drawbacks of matrix factorization

- only encode user and item standalone information
	- missing lots of interactions and external information
- cold start problem
	- must have samples to learn latent variables
	- cannot deal with frequent new items
		- news sentiment

regression-based model

- view latent variables as a linear transformation from explicit features (dimension reduction)
	- the transformation serve as a prior and will be changed by new samples
- train latent variables by matrix factorization
- regress between latent variable to explicit features 
- a hierarchical Bayesian model
	- more proper training would be Expectation Maximization
	- remediated the cold-start problem

63 

Factorization Machine

- regession based factorization is very hard to train

Steffen Rendle. Factorization Machines with libFM. ACM Trans. Intell. Syst. Technol. 3, 3, Article 57 (May 2012), 22 pages, 2012.

1. create all user and item features
2. create interaction features with basic features (eg. product of two features), build a factorization model based on products 
    1. every features has one latent variable vector, the product of two features is the dot product of two latent variable vectors
    2. learn factorization of the matrix to avoid high-dimension problem
        1. train with SGD is ok

在最近几年的 Kaggle 比赛中以及一些工业级的应用中，分解机凭借其简单易用的特点，成为了很多产品的核心算法。

64

Tensor Factorization

- matrix can only incorporate one-one relationship, lose context
	- eg, location and time

tensor can incorporate N-N relationship

- CP factorization(CANDECOMP/PARAFAC)
	- decompose N dimension tensor to N m matrices
		- eg. NxMxR to NxK, MxK, RxK 
		- K is the dimension of hidden variables, same across matrices
- HOSVD (High Order Singular Value decomposition)
	- decompose to different N matrix plus one small tensor
	- different hidden variable dimensions across matrices
- training
	- need to design loss function
		- eg. squared loss
	- use SGD
	- use Alternating Least Seuqare

1. Alexandros Karatzoglou, Xavier Amatriain, Linas Baltrunas, and Nuria Oliver. Multiverse recommendation: n-dimensional tensor factorization for context-aware collaborative filtering. Proceedings of the fourth ACM conference on Recommender systems (RecSys '10). ACM, New York, NY, USA, 79-86, 2010.

https://xamat.github.io/pubs/karatzoglu-recsys-2010.pdf

65

Collective Matrix Factorization

- use matrix to model every pair of one-one relationship
- assume the hidden variables by factorization are the same
	- to link different matrices
	- project different matrices to same user and item space
- training
	- use SGD to train

drawbacks

- strong assumptions
- data amount differs
	- user-item data more than user-user data

1.  Ajit P. Singh and Geoffrey J. Gordon. Relational learning via collective matrix factorization. Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '08). ACM, New York, NY, USA, 650-658, 2008.

http://www.cs.cmu.edu/~ggordon/singh-gordon-kdd-factorization.pdf


66 objective function design

- rating based or click-through or purchasing rate based
	- linear regression
	- poisson regression 
- rank order based : more realistic
	- BPR method: pairing method
		- rank positive interaction before negative interaction
		- implicit feedback: assume ones with positive feedbacks more related than ones the user has not seen
	- related: optimize NDCG, MAP 
		- combine with factorization, factorization machine to build model


 Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. BPR: Bayesian personalized ranking from implicit feedback. Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence (UAI '09). AUAI Press, Arlington, Virginia, United States, 452-461, 2009.

67 

Exploitation - Exploration  Method

Yahoo Today Module data

- relatively concentrated high-quality resource
	- 100 articles in content pool
- large amount of users
	- convergence is almost guraranteed
	- unbiased data collection
	- epsilon greedy

Deployment is harder than algo selection
	
- online testing
	- same user should have same experience - must distinguish visits from different users
	- bucketing based on user - some user might be sacrificed
- how to balance between products
	- some users sacrifice experience for other users
- improvement
	- only test on new users
	- always have high-quality content, ranked before new content

68 
Upper Confidence Bound Algorithm

Epsilon-Greedy

- with probability P  display items with current estimation
- with probability 1-P dis display randomly choose all items

UCB

- deterministic algo
- empirical mean + 2 standard deviation as score for ranking items
- with more rounds, converge to true mean

drawbacks

- many items with 0 mean (never been displayed)
- deterministic in natrue

69

Thompson Sampling

- build Bayesian distribution model
	- eg. click-through as a Bernoulli distribution + conjugate Beta distribution
	- can have prior
- sampling from posterior parameter distribution 
- repeat, probability distribution converges
- stochastic algo

drawbacks

- can be complicated and requires modeling
	- MCMC, importance sampling


70

model evaluation 

offline evaluation

- mean square error
	- biased towards frequent users
	- ignored relationship between items
- rank order based
	- precision & recall (AUC) of related items extracted/recommended
	- NDCG, MAP, etc

drawbacks

- ranked items with feedbacks only



71

online model evaluation

AB Testing (Causal inference, conterfactuals ***) online

independent variable to dependent variables

- treatment bucket 
- control bucker

could be hard

- confounding factors, conterfactuals
- unknown factor, hard to randomize
- cannot AB testing on only variable on variables in realities (different modules of website)

online evaluation - use online measures

- Dwell time
	- 用户粘性
- Absence time


73

unbiased estimation (bias corrections)

- clicked vs unclicked
- time, position, etc


- combine with data collection (EE method)
	- tradeoff: user experience vs display more items
- weighted sampling
	- users clicked unfrequent items(less probability to show), give more weights
		- inverse proportional correction



74

architeacture

1. fast recommendation 100ms - 200ms
2. react to interactions
    1. not display unliked items
3. cover all clients, sclable

offline vs online

学术界的推荐系统，或者是 Kaggle 竞赛的推荐系统，往往是一个使用了很多不同模型的集成模型（Ensemble Model），这种方式虽然在比赛和论文发表中能够取得较高的精度，但是在现实的系统中，如果不加修改直接使用，必然无法在规定的时间内，也就是一两百毫秒内产生所有的推荐结果。同样的，很多非常复杂的深度学习模型，也无法在规定的时间内产生所有的推荐结果。由此可见，很多推荐架构的核心就是在解决这些问题。


- calculate offline, store results in databse
- drawbacks
	- unable to handle new items
	- not suitable for frequent items(News)


75

multi-layer search architecture

- first scoring: extract relevant items (use linear or boolean functions) - build index
- second-round scoring: rank - tree-based models

1. fast recommendation : by index and first layer
2. interaction update second layer features
    1. re-order： depends on complexity
    2. update some features only
3. new users
    1. use old user data ranking (as prior)
4. new items - drawbacks
    1. rebuild index could be hard


76 modren architecture

- new user problem
	- get feedbacks from user fast, react fast
		- update features (multi-layer, dynamic architecture)
	- lack of data: cold-start
		- age, gender, location information -> get similar users (proxy) update frequently
- new item problem - harder
	- new content pool -> add to all users : time consuming
	- pre-scoring (similar items or random score ) -> mix with old items
	- indexing
		- update index : must do offline, time consuming
		- build temporary index, combine with old index










—————————————————

深入学习follow-up


核心 

get more informaiton

- financial machine learning vs machine learning in 搜索，推荐

- 贝叶斯prior set, (uninformed)
- smoothing in finance and other estimates(summary)
	- EWMA
	- bayesian
	- kalman filtering
- collaborative filtering in factor investing
- similarity, clustering and measures
- good formulation examples(***- supervised, unsupervised)

极客时间 人工智能


十大算法

SVM

http://www.hyadatalab.com/papers/analogy-kdd17.pdf



deep learning

RNN

- GRU
- LTCM

高级数据挖掘

http://www.hyadatalab.com/papers/analogy-kdd17.pdf


知识图谱


Engineering


http://delivery.acm.org/10.1145/3100000/3098026/p1507-hou.pdf?ip=104.245.8.202&id=3098026&acc=OPENTOC&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E054E54E275136550&CFID=824613284&CFTOKEN=25201339&__acm__=1509500476_9d244f060207e966c107eb505646ed55


——————

2020 前沿

机器学习方法
- end to end learning
- bias
	- discriminarion
	- unbalanced data
- NP hard
	- 非凸优化
	- 分不建议
	- 非对称博弈
	- 排序模型
- meta learning
- 框架建议
- 搜索
- bias
- causual inference
	- Thorsten 第三个主要学术贡献，也是他最近几年的学术成功，那就是把因果推论（Causal Inference）和机器学习相结合，从而能够更加无偏差地训练模型。可以说这部分工作开创了一个新领域。长期以来，如何有效地应用用户产生的交互数据来进行模型训练，都是大规模机器学习特别是工业界机器学习的难点。一方面，工业系统能够产生很多用户数据；另一方面，这些用户数据又受到当前部署系统的影响，一般都有一定的偏差。因此工业级机器学习系统面临一个长期挑战，那就是，如何能够在评估模型以及训练模型的时候考虑到这样的偏差，从而去除这样的偏差。Thorsten 利用因果推论中的倾向评分（Propensity Scoring）技术以及多臂赌博机（Multi-armed Bandit）思想，把这样的方法成功地引入到机器学习中，使得无偏差地训练模型成为可能。目前，这方面的新研究和新思想正在机器学习以及应用界产生越来越多的共鸣。

TODO

- 下载经典论文

信息源

- 量子位
- 机器之心
- 


读法

1. 作者群 公司 机构
2. 核心略读
3. 具体技巧
4. 