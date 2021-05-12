# 广告系统

## 核心技术

### 平台

* SSP-to publishers/media
- DSP- advertisers
- ADX - match SSP, DSP & bidders
- DMP - data management platform

### 技术

* 追踪用户\*
	- Cookie 的整合。
		* B 网站直接得到 A 网站的允许，到 A 网址植入脚本从而来收取用户的 Cookie 信息。例如，在某个时期内，纽约时报的网站就有多达 17 个合作方在对用户的数据进行收集。然而，即便是这样，每个单独的数据收集方都只能对用户在互联网上的行为进行局部的数据采集。也就是说，这些收集方很难对用户在互联网上的全部行为进行建模。很明显，这是不利于展示最有价值的广告信息的。
		* 简单说来，Cookie 整合要做的事情就是把多个由不同的数据收集方收集的 Cookie 进行匹配，从而能够在逻辑上把这些 Cookie 都当做同一个人处理。据统计，一个用户在 30 次点击内，就有 99% 的概率会被互联网上前 10 大“数据追踪机构”所追踪，而至少有 40% 的 Cookie 可以得到有效的整合。
		- 用 Cookie 来追踪用户并不是万能的。用户可以删除 Cookie 信息甚至在浏览器的层面禁止 Cookie 信息。这就给广告平台提出了不小的挑战。最近几年，基于其他存储技术的用户追踪手段，例如 Canvas API 或者 Flash Cookie 等也慢慢流行起来。
* 广告回馈预估feedback estimation
	- click, convert（点击，购买，交易，etc）
		* (user, content, ads) - 素材，场景，广告商等因素
		- 某一种类型的广告在哪个发布商能够带来更多的点击，从而能够有针对性地对于某个发布商进行投放
	- 数据-在广告点击率预估的问题中，正例的数目常常是负例的百分之一或者千分之一。这样造成的就是非常“不均衡”的数据集。
		- 第一，上下文的特性信息非常重要。这两个广告可能是类型不同，可能展示的地区不同，因此并不能完全直接来对这两个广告进行比较。
		* 第二，广告 2 在旧金山地区的展示次数还比较少，因此 0.03 这个预估值可能是非常不准确的，或者说至少是不稳定的，它的误差要大于第一个广告。
		- 平滑
			- yahoo研发人员提出了一个点击率估计方法，其实也是一种两层模型。第一层模型就是最原始的对点击率的估计，也就是类似我们上面所说的直接按照数据进行估计。当然，这里的问题我们刚才也已经提到了，就是估计的不稳定性。第二层模型是对第一层模型的修正。所谓修正，就是利用层次化信息来对原始的估计值进行“平滑”（Smoothing）。
			- 两个广告来自于同一个广告商，因此它们应该有一定的类似的点击率；两个广告被展示到同一个地区，它们也应该有一定的类似的点击率。这些层次信息给了我们一些启示，来对原始估计值进行修正。当然，根据我们这两个例子你就可以看出，一个广告可以受到多个层次信息的影响，比如广告商的层次信息，地理位置的层次信息，以及类别的层次信息等。所以，要想设计一套完善的基于层次信息的平滑方案也并非易事。
			- 雅虎在这方面的工作都围绕着一个主题，那就是如何对平滑方案进行创新。一种方法是利用“产生式模型”（Generative Model）的概念，把层次信息的叶子节点的数据产生过程，定义为基于其父节点数据的一个概率分布产生过程，从而把整个平滑方案的问题转换成为了一个有向无环图上的每个节点的后验概率参数的估计问题（参考文献[1]和[2]）。
			* 另外一种方法则采取了一个不太一样的思路，那就是在做平滑的时候，在这种产生式建模之后，还追加了一个过程，利用树模型来对平滑的结果进行再次修正，使得最后的结果能够达到更高的精度（参考文献[3]）。
	- 目标函数 - 比搜索和推荐复杂
		- 真实的系统中，我们需要在很多候选可能的广告中，选出最优的一个或者几个显示在页面上。从某种程度上来说，这更像是一个排序问题。
		* 同时，对于不少 DSP（需求侧平台）来说，广告排序的最终目的是进行“竞拍”（Auction）。因此，最后估算广告的点击率以后，还需要看广告的竞价，由此来对广告是否会赢得竞拍从而被显示在页面上进行一个更加全面的估计。
	- 算法
		- 二分类分析- logistic regression
			- 普通的逻辑回归并不适应大规模的广告点击率预估。有两个原因，
				* 第一，数据量太大。传统的逻辑回归参数训练过程都依靠牛顿法（Newton’s Method）或者 L-BFGS 等算法。这些算法并不太容易在大规模数据上得以处理。
				* 第二，不太容易得到比较稀疏（Sparse）的答案（Solution）。
			- FTRL - follow the regularized leader 参数在每一个数据点更新
				- 第一部分是一个用过去所有的梯度值（Gradients）来重权（Re-Weight）所有的参数值；
				* 第二部分是当前最新的参数值尽可能不偏差之前所有的参数值；
				* 第三个部分则是希望当前的参数值能够有稀疏的解（通过 L1 来直接约束）
			- 另外一个比较新颖的地方，就是对每一个特征维度的学习速率都有一个动态的自动调整。传统的随机梯度下降（Stochastic Gradient Descent）算法或是简单的在线逻辑回归都没有这样的能力，造成了传统的算法需要花很长时间来手工调学习速率等参数
				- 而 FTRL 带来的则是对每一个维度特征的动态学习速率，一举解决了手动调整学习算法的学习速率问题。简单说来，学习速率就是根据每一个维度目前所有梯度的平方和的倒数进行调整，这个平方和越大，则学习速率越慢。
			- 调优
				- 文章介绍了利用布隆过滤器（Bloom Filter）的方法，来动态决定某一个特征是否需要加入到模型中。虽然这样的方法是概率性的，意思是说，某一个特征即便可能小于某一个值，也有可能被错误加入，但是发生这样事件的概率是比较小的。通过布隆过滤器调优之后，模型的 AUC 仅仅降低了 0.008%，但是内存的消耗却减少了 60% 之多，可见很多特征仅仅存在于少量的数据中。
				- 来减少内存的消耗。比如利用更加紧凑的存储格式，而不是简单的 32 位或者 64 位的浮点数存储。作者们利用了一种叫 q2.13 的格式，更加紧凑地存储节省了另外 75% 的内存空间。
				- 每一步 FTRL 更新的时候，原则上都需要存储过去所有的梯度信息以及梯度的平方和的信息。文章介绍了一种非常粗略的估计形式，使得这些信息可以不必完全存储，让内存的消耗进一步降低
			- Calibration
				* 文章也提出了需要对模型的最后预测进行调优（Calibration），使得模型的输出可以和历史的真实点击率分布相近。这一点对于利用点击率来进行计费显得尤为重要，因为有可能因为系统性的偏差，预测的数值整体高出或者整体低于历史观测值，从而对广告主过多计费或者过少计费。
			- 失败
				- hashing trick 特征哈希降低内存
				- dropout, normalization
		- tree models/GBDT
			- 点击率预估模型分为两个层次。也就是说，从最初的模型特性输入，需要经过两个不同的模型才对点击率做出最终的预测。这个两层架构对后来的很多点击率预估模型有巨大的影响。
				- 连续数值的特性已经被转换成了离散的数值。然后，这些离散的数值经过了一个 GBDT 树来进行特性转换
					- 第一，GBDT 可以对特性进行非线性组合。也就是说，GBDT 的输出一定是之前特性的非线性的转换，这是由树模型原本的性质所带来的，这个性质对于线性模型来说会有巨大的优势。
					- 第二，经过 GBDT 转换之后，树模型其实选择出了对目标有用的特性，因此这里还起到一个“特性筛选”（Feature Selection）的作用。也就是说，经过 GBDT 的模型，最后剩下的特性肯定是要远小于最初的输入特性的，毕竟有作用的特性是少数的
				- 经过了 GBDT 之后，Facebook 的研究者用树模型最后的叶节点当做新的特性，然后再学习了一个线性的分类模型。这思想其实和后来流行的深度学习的想法很类似，也就是先对输入特性进行非线性转换，然后再经过一个线性分类器来进行最后的预测。这个第二层的线性分类器可以用类似 SGD 的方法进行“在线学习”（Online Learning）。
				- 不仅展示了两层模型的优势，并且还讨论了很多选取特性方面的经验以及训练模型的经验，比如广告过去的历史信息非常重要，而绝大多数重要的特性都和历史信息有关。
					- 
		- 深度模型
	- 模型评估
		- 比较通行的评测不均衡数据分类问题的指标是“曲线下面积”，或者简称为 AUC，这个评测办法可以算是一种替代方法。简单来说，AUC 就是看我们是不是能够把正例给排序到负例上面。也就是说，如果每一个正例和负例都有一个预测数值，那么我们按照这个数值排序，去数每一个正例下面有多少负例，然后对所有正例所对应的数取平均。AUC 的数值高，则代表我们可以把绝大多数正例排序到负例前面。
		- AUC 的一个最大问题就是它并不在乎所有实例的绝对预测数值，而只在乎它们的相对位置。这在广告系统中可以说是一个非常大的缺陷。我们之前也提过，有很多广告系统组件依赖于对于广告点击率的精确预估，比如收费系统，流量预测等。因此，仅有一个相对位置的正确是不够的。  
		- “归一化的交叉熵”，简称 NE，用于衡量广告系统的好坏。NE 实际上是一个比值，比值的分母是数据中观测到的实际的点击率的数值，也可以叫作数据的“背景估计”（Background Estimation）；而分子是某一个模型对点击率的估计。这样做的归一化，目的就是来看，在去除了背景估计的情况下，对点击率的估计是否依然好或者坏。
* 出价策略，竞价
	- generalized second price auction 第二价位竞拍
		- 在基于第二价位竞拍的形式下，广告商按照自己对于广告位价值的理解来竞拍是相对较优的策略。
		- Jun Wang, Weinan Zhang and Shuai Yuan. Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting. Foundations and Trends® in Information Retrieval: Vol. 11: No. 4-5, pp 297-435, 2017.
	- Bidding Landscape predictions - 对于“竞价全景观”或者是赢的价格分布的估计有一个比较困难的地方，那就是，作为广告商来说，往往并不知道所有其他竞争对手的出价，以及在没有赢得竞拍的情况下，那些赢得竞拍的出价是多少。简而言之，也就是我们只观测到了一部分数据，那就是我们赢得这些广告位的出价。在这种只有一部分信息的情况下，所做的估计就会不
		* Wu, W. C.-H., Yeh, M.-Y., and Chen, M.-S. Predicting winning price in real time bidding with censored data. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1305–1314. ACM, 2015.
		- 有学者认为，赢的价格服从一个“对数正态分布”（Log-normal）。也就是说，广告商出的价格并且最终赢得竞拍的这些价格，在取了对数的情况下，服从一个正态分布。
		- 利用了一种对数几率回归来估计那些没有赢得竞拍情况下的赢的价格，然后和已知的赢的价格一起对整个“竞价全景观”进行估计，这也算是目前的一项前沿研究。
	- bidding strategies
		- 竞价的一个重要特征，就是作为一个竞标方，我们并不知道其他竞标方的点击率和出价。因此，我们处在一个信息不完整的竞价环境中。在这样的环境中，我们只能根据自己的点击率估计和自己的出价，以及过去出价的成功与否来对整个市场的形势进行判断。这就是在 RTB 中竞价策略的一大挑战和难点。
		- 思路是把整个竞价策略当做一种“博弈”（Game），从而根据博弈论中的方法来对竞价环境中各个竞标方的行为和收益进行研究（比较经典的论文例如参考文献[1]）。用博弈论的方法来对竞价进行研究有一个最大的特点，那就是博弈论主要是对各个竞标方行为之间的关联性进行建模，这种关联性包括他们之间的收益和他们的动机。
			- 也就是利用博弈论的方法来对竞价策略进行研究主要存在于学术界。虽然从理论上来说，博弈论可能提供一种比较有说服力的解释，但是这种思路需要对整个竞价环境有非常多的假设（例如竞标方是不是理性，市场是不是充分竞争等等）
		- 另外一种思路是把整个竞价策略当做是纯粹的统计决策，也就是直接对广告商的行为进行建模，而把整个竞价环境中的种种联系都当做是当前决策下的不确定因素（这种思路比较有代表性的论文是参考文献[2]）。在这样的思路下，各个竞标方之间的行为关联变得不那么重要，而对于整个不确定性的建模则变得至关重要。
			- 而第二种思路，仅仅需要从广告商自身的角度出发，因此在现实中，这种思路的操作性更强，从而受到工业界的青睐。
			- 第二种思路其实就是根据当前的输入信息，例如页面信息、广告信息、用户信息以及上下文信息等，学到一个输出价格的函数，也就是说，这个函数的输出就是在现在情况下当前广告的出价。当然，这个函数势必需要考虑各种不确定的因素。
		- 搜索vs展示
			- 对于搜索广告来讲，在大多数情况下，每一个出价都是针对某一个搜索关键词的
			- 参考文献[3]是第一个利用机器学习方法对搜索广告的出价进行建模的工作。在这个工作里，每一个关键词的出价来自于一个线性函数的输出，而这个线性函数是把用户信息、关键词以及其他的页面信息当做特性，学习了一个从特性到出价的线性关系。
			- 展示广告的竞价则面临着不同的挑战。首先，在展示广告中，场景中并不存在搜索关键词这种概念。因此，很多广告商无法针对场景事先产生出价。这也就要求 RTB 的提供商要能够在不同的场景中帮助广告商进行出价。
			- 相比于搜索广告针对每一个关键词的出价方式来说，针对每一个页面显示机会出价的挑战则更大。理论上讲，每一个页面显示机会的价格都可能有很大的不同。很多 RTB 都利用一种叫作 CPM 的收费模式，也就是说，一旦某一个广告位被赢得之后，对于广告商来说，这往往就意味着需要被收取费用。所以，在展示广告的情况下，如何针对当前的页面显示机会以及目前的预算剩余等等因素进行统一建模，就成为一个必不可少的步骤。
		- Pacing
			- 一个广告商现在有 1 千元的预算参与到 RTB 竞价中。从广告商的角度来说，通常希望这 1 千元能够比较均匀地使用到整个广告竞价中。或者说，即便不是完全均匀使用，至少也不希望这笔预算被很快用完。这里面的一个原因是，在每天的各个时段，广告的表现情况，也就是说转化率或点击率是不一样的，广告商通常希望自己的广告能够在比较好的时段进行展示
			- 告竞价策略中，还存在着一个叫“预算步调”（Budget Pacing）的技术，也就是希望能够让广告的展示相对平缓而不至于在短时间内使用完全部的预算。这势必对于广告如何出价有着直接的影响。
		- 对于平台而言，虽然竞价保证了一定的竞争，但是也并不是所有的展示机会都有非常充分的竞争。因此，从平台的角度来说，如何能够保证一定的收益就变得十分重要。在这样的情况下，有的平台有一种叫作“保留价格”（Reserved Price）的做法，用来设置一个最低的竞价价格。保留价格虽然能够来保证收益，但是也可能会让广告商觉得不划算，因此如何来设置这个保留价格，也就成为了出价策略中的一个重要组成部分。
		* optimal bidding
		- campaign level bidding optimization
			- 这里我们采用一个简化的假设，认为一个推广计划的出价是点击率的一个函数。
				- 第一个概念是“赢的概率”（Winning Probability）。这里面，如果我们知道现在市场的一个价格分布以及我们的出价。那么，赢的概率就是一个已知概率密度函数求概率的计算，也就是通常情况下的一个积分计算。
				* 第二个概念就是“效用”（Utility）。这是一个广告商关注的指标，通常情况下是点击率的某种函数，比如利润，那就是每一次点击后的价值减去成本。
					- 成本其实主要就是出价后产生的交易价格。如果是基于第一价位的竞价，那么这个成本就是出价；如果是基于第二价位的竞价，这个成本就是超过第二价位多少还能赢得竞价的价格。
				- 所有的广告推广计划都必须要在预算内，这是一个很明显的限制条件。
			- 假设，我们竞价的核心是所谓的“按照价值”的竞价。那么，在这种情况下，最优的策略其实就是按照点击率乘以点击后产生的价值来进行出价。可以说，这种策略其实是业界接纳程度最好、也是最直观的一种竞价策略。
			- 了预算和当前的交易流量信息的情况下，这种竞价策略就并不是最优的策略了。为什么呢？因为在有了这些限制条件的情况下，我们是否还会按照自己客观认为的广告价值来竞标就成了一个疑问。
				- 线性出价[1]与其完全按照广告的价值来进行出价，不如采用这个价值乘以某个系数，而利用这个系数来动态调整目前的出价。由于是在一个已知的可能出价前面乘以一个系数，所以整个出价策略其实是一种线性变换，因此也被叫作是线性出价策略。
					- 在这篇论文中，这种算法也取得了比较好的效果。不过遗憾的是，这种做法并没有太多的理论支持。
				- 【2】【3】这个框架的整体思路是把寻找最优出价，或者说是竞价函数的过程表达成为一个“有限制的最优化问题”（Constrained Optimization）。
					* 最优化的优化目标，自然就是当前竞价流量下的收益。而最优化的限制条件，就是竞价流量下的成本要等于预算。也就是说，在我们期望达到预算的情况下，我们需要尽可能地扩大收益，这就是最优化目标的最大化这个意思。而限制条件决定了这个最大化问题的解的空间，因此，那些不符合条件的解就可以不考虑了。
					* 一旦我们的问题可以用有限制的最优化问题来表达以后，整个问题的求解就变得相对比较规范化了。对于这类问题有一个标准的求解过程，就是利用“拉格朗日乘数法”，把“有限制的优化问题”转换成为“无限制的优化问题”，然后针对最后的目标函数，求导并置零从而推导出最优解的结果。这一部分的步骤是标准的高等数学微积分的内容。
					* 这个框架最后推导出了基于第一价位和基于第二价位的最优的出价函数形式。在两种情况下，最优的出价函数都是一个基于点击率、当前竞价流量和预算的非线性函数。那么，从这个框架来看，刚才我们提到的线性竞价策略就并不是最优的。
		- 多个计划优化
			- 在这方面比较经典的论文，推荐你读一读《展示广告的统计套利挖掘》（Statistical Arbitrage Mining for Display Advertising）[4]。从基本的思路上来讲，我们需要做的是把刚才的基于单个广告推广计划的有限制优化问题给扩展到多个广告推广计划上去。除了满足各种限制条件以外（比如需要满足总的预算要求），论文也提出了一种基于风险控制的思路，来计算每一个广告推广计划的均值和方差，从而限制方差的大小来降低风险。
			* 比较遗憾的是，论文提出的优化是一个基于 EM 算法的过程，也就是说相对于单个广告推广计划来说，多个广告推广计划找到的解可能并不是全局的最优解。（***\)*
* Budget Pacing
	- 在每一个时段，发布商所面临的受众都有可能不太一样，所以，对于广告商而言，比较理想的状态是一个广告可以在一天的不同时段被不同的受众所看到，从而达到扩大受众面的目的。
	- 一种叫“节流”（Throttling）
		- 节流这种方法主要是把单位时间的支出或者是成本给控制在某一个速率内，使得预算能够被均匀地使用。这种方法往往是在我们已经介绍过的竞价模型之外运行。
		- 节流思路，有一种做法[1]是把如何节流当做一种“线性优化”问题，并且是有限制的最大化问题。具体来说，对于每一个出价的请求，我们都可以做一个二元的决定，决定我们是否接受这个出价请求。当然，对于每一个出价请求，这里都有一个价值和一个成本。根据对不同出价请求的设置，我们来做优化，从而能够最大化总价值。但同时，我们需要遵守一个限制，总的成本不能超过预算。这其实就是在两种目标之间实现一个均衡，简言之，我们需要在不超过总预算的情况下达到总价值的最大化。虽然这种算法本身能够通过我们之前介绍过的“拉格朗日乘数法”来求解，
		* 但是还存在一个根本的问题，那就是这种算法并不能实时地对整个竞价的安排进行计算和更新。因为，这种线性优化方法一般都是在线下计算好了以后再到线上运行。很明显，这种方法并不适合快速变化的竞价环境。
		* 也就有一些工作[2]和[3]，尝试通过节流，或者更确切地说，通过在线优化来控制预算的使用情况。
	* 一种叫“修改出价”。
		- 修改出价这个思路很直观，也就是直接修改我们的竞价，从而带来预算均匀化的结果。
		- 竞价直接进行修改的相关工作也很多[4]和[5]，这个思路是把控制理论中的一些思想借鉴到了对竞价的直接优化上，目标是让广告商的预算能够平滑使用。这里面的控制是指什么呢？主要是指我们引入一个新的模块在 DSP 中，从而能够实时监测各种指标，例如竞价赢的比率、点击率等，然后利用这些数据作为一个参考点，从而能够形成一种回馈信息以供控制系统来对出价进行实时的调整。和节流的思想相对比，利用控制理论对出价进行直接优化这种思路明显要更加灵活。
		* 然而在实际的工作中，更加灵活的框架依赖于对点击率以及竞价全景观的准确预测，这其实是很困难的。在真实的情况下，利用节流的思想，也就是不去修改出价，只是在其基础上直接进行操作，则往往来得简单有效。
	- 频控
		- 在工业界，还有一种经常会使用的控制预算的方法叫“频率上限”（Frequency Cap）。简单来说，这种策略就是限制某一个或者某一种广告在某一种媒介上一段时间内出现的次数。比如，是否限制一个肯德基的广告在半天之内让同一个用户看见的次数？5 次、10 次还是 20 次？为什么要限制频率呢？
		* 一个因素当然是我们希望广告的预算不要在短时间内消耗完。另外，短时间内反复观看某一个广告，很可能会让用户对某一个广告或者广告商产生厌烦情绪，那么广告的有效度就会降低。这对于一些广告商来说，其实是消耗了一些资源。因此，限制广告的投放是一种策略选择，从而让广告的投放花钱少、效率高。
		* 这种频率上限的做法在工业界非常普遍，不过比较遗憾的是，关于这样做究竟是不是有很大的效果，用户多次看到广告是否会真正产生非常大的厌烦情绪从而使得广告效果降低，有没有理论支持等问题，目前还没有比较好的研究来解决。
	- Reserved Price

## 论文

* “Ad Click Prediction: a View from the Trenches”） - Google
	* KDD 2013 年的工业论文组，在短短几年时间里就获得了近 200 次的文章引用数，不少公司争相研究其中的内容，希望能够复制类似的算法和技术。
		- 外一位作者斯卡利（D. Sculley）从塔夫茨大学（Tufts University）博士毕业之后，一直在 Google 的匹兹堡分部工作，并着手研究大规模机器学习系统，其中重要的代表性研究成果是如何把回归问题和排序问题结合起来（发表于 KDD 2010 年）
			- 大规模版本的Rank SVm
	- 这篇文章提出了用一种叫 FTRL（Follow The Regularized Leader）的在线逻辑回归算法来解决上述问题。FTRL 是一种在线算法，因此算法的核心就是模型的参数会在每一个数据点进行更新。FTRL 把传统的逻辑回归的目标函数进行了改写。
- 2014 年发表的一篇论文《Facebook 的广告点击率预估实践经验》（Practical Lessons from Predicting Clicks on Ads at Facebook）
	- Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, and Joaquin Quiñonero Candela. Practical Lessons from Predicting Clicks on Ads at Facebook. Proceedings of the Eighth International Workshop on Data Mining for Online Advertising (ADKDD’14). ACM, New York, NY, USA, , Article 5 , 9 pages, 2014.
- 雅虎
	- Deepak Agarwal, Andrei Zary Broder, Deepayan Chakrabarti, Dejan Diklic, Vanja Josifovski, and Mayssam Sayyadian. Estimating rates of rare events at multiple resolutions. Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '07). ACM, New York, NY, USA, 16-25, 2007.2. Deepak Agarwal, Rahul Agrawal, Rajiv Khanna, and Nagaraj Kota. Estimating rates of rare events with multiple hierarchies through scalable log-linear models. Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '10). ACM, New York, NY, USA, 213-222, 2010.3. Nagaraj Kota and Deepak Agarwal.Temporal multi-hierarchy smoothing for estimating rates of rare events. Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '11). ACM, New York, NY, USA, 1361-1369, 2011.4. Olivier Chapelle, Eren Manavoglu, and Romer Rosales. Simple and Scalable Response Prediction for Display Advertis
		- 列工作虽然在概念上有很高的学术和实践价值，特别是如何利用层次性信息来对预测进行平滑这个方面，但是从整体来说，预估方案变得非常复杂而且环节太多。
	- 雅虎后期的广告预估模型又从比较复杂的两层模式转换为了一层模式。这个转换主要是考虑到了整个流水线（Pipeline）的复杂度以及需要处理的数据规模逐渐变大，那么利用更加稳定和简单的方法就势在必行了。对于雅虎后期的广告预估模型，我参考论文《简单和可扩展的展示广告响应预测》（Simple and Scalable Response Prediction for Display Advertising）[4]，在这里为你简单做一个总结。
	- 总体来说，整个模型回到了相对简单的“对数几率回归”（Logistic Regression），并且直接对所有的特性（Feature）进行建模。这里面唯一可能和之前的很多工作不太一样的地方，是大量使用了“特性哈希”（Feature Hashing）的方法。简单来说，特性哈希就是把原来大规模的有可能是极其稀疏的特性给压缩到了一个固定维度的特性空间里。当然，这肯定会对精度等性能有一定影响，因此这是一个需要有一定取舍的决策。
		- 篇论文中，作者们还介绍了如何对大量的数据进行采样，以及如何利用配对的特性（也就是把两种不同的特性，比如广告商和地理位置进行配对）来自动产生更多的非线性因素的方法。那么这个一层模式的方法所达到的效果怎样呢？论文中论述，相比于之前的两层结构，这个方法所达到的效果有很大程度的提升。
- LinkedIn
	- LinkedIn 的广告预估模型。这个模型的一大“卖点”就是直接充分考虑了“冷启动”和“热启动”两种模式。
	- 于“冷启动”，“热启动”指的是我们已经掌握了用户或者广告的一定信息，然后利用这些历史信息来对点击率进行预测。这么说来，我们一般需要有两套对策，一套针对“冷启动”，一套针对“热启动”。LinkedIn 的方法就是希望通过一个模型来同时解决这两个问题。
		- Deepak Agarwal, Bo Long, Jonathan Traupman, Doris Xin, and Liang Zhang. LASER: a scalable response prediction platform for online advertising. Proceedings of the 7th ACM international conference on Web search and data mining (WSDM '14). ACM, New York, NY, USA, 173-182, 2014.
		- 第一部分，是利用用户、广告和上下文所建立的全局性预测。什么意思呢？就是我们利用用户特性、广告特性以及上下文特性来对点击率进行预测。这部分的核心思路就是这些特性所对应的系数是全局性的。也就是说，对于不同的用户、不同的广告以及不同的上下文所对应的系数是相同的。因为是全局性的系数，因此这部分其实提供了一种“冷启动”的需求，也就是不管是任何新的用户或是广告，只要有一定的特性，我们总能通过这部分得到一种粗略的估计。
		* 第二部分，是利用第一部分的用户、广告和上下文信息组成交叉特性，从而学习这些特性之间的关系。如果说第一部分直接就是线性的预测，那么第二部分其实就是“交叉项”形成的非线性的部分。我们之前在讲推荐系统的时候提到过“分解机”（Factorization Machines）这个模型，讲到过这种“交叉项”所带来的非线性预测的好处。虽然这里和分解机的构成不完全一样，但是整体上表达了相似的意思。
		* 第三部分，是 LinkedIn 模型提出来的独特之处（和其他公司模型不太一样的地方）。那就是同样是利用用户、广告和上下文特性，但是 LinkedIn 所提模型的系数则是每个用户、广告和上下文都不同。作者们认为这可以实现“热启动”效果。也就是说，当某个用户、某个广告或者某个上下文已经有比较多的数据以后，就可以依靠这些用户、广告或者上下文自己的系数了，而不仅仅依靠第一部分的全局系数。这个第三部分只有当数据比较多的时候才能够起作用。
		- 作者们认为，刚才模型中所说的三个部分所需要的模型更新频率是不一样的。
		* 比如第一部分和第二部分都可以认为是全局模型，也就是说系数是全局性的。因此这些模型的变化会比较慢，作者们建议一个星期对模型进行一次更新。
		* 而第三部分则是在已经积累了历史信息后慢慢呈现出的效果，因此对于数据会非常敏感，而且每个用户和每个广告都是不同的系数，因此需要在短时间内，比如半个小时甚至几分钟内，就重新训练模型，以达到个性化的目的。
		- 提出的模型和 EE（Exploit & Explore）策略结合了起来。我们在讲推荐系统时介绍过 EE 的思路，简单回顾一下 EE 的目的，主要就是探索那些并没有太多机会被展示的物品，在这里也就是广告。我们刚才说了，所有的系数都加上了先验概率，因此其实可以很容易结合数据计算后验概率分布。有了后验概率分布，作者们提出了以汤普森采样为主的 EE 模式。这也可以算是论文提出模型的一大亮点。
		- 大规模的数据上对模型进行训练，这篇文章采用了一种 ADMM 算法。在文章提出来的时候，作者们还是希望能够利用单个服务器对所有的模型参数进行训练。和其他的算法相比，一般认为 ADMM 这种算法的收敛速度更快，但是，利用这种算法的其他公司并不太多。
- Twitter
	- 《Twitter 时间轴上的广告点击率预估》（Click-through Prediction for Advertising in Twitter Timeline）
		- 的 Facebook 的案例，其实也是往用户的信息流中插入广告。很多类似的社交媒体都争先恐后地开始进行相似的项目，这一类广告经常被称为社交广告。社交广告的特点是，需要根据用户的社交圈子以及这些社交圈所产生的内容，而动态产生广告的内容。广告
		- 抖音社交？
		- 排序学习
			- 首先，排序学习中最基本的就是“单点法”（Pointwise）排序学习。回顾一下，单点法其实就是把排序学习的任务转化为分类问题。其实典型的就是直接利用“支持向量机”（SVM）或者对数几率回归模型。
			* 第二种比较常用的排序学习的方法就是“配对法”（Pairwise）排序学习。通俗地讲，配对法排序学习的核心就是学习哪些广告需要排到哪些广告之前。这种二元关系是根据一组一组的配对来体现的。学习的算法，主要是看能否正确学习这些配对的关系，从而实现整个排序正确的目的。对于配对法排序，我们依然可以使用对数几率回归。
			* 只是这个时候，我们针对的正负示例变成了某个广告比某个广告排名靠前，或者靠后。值得一提的是，通过配对法学习排序学习，对于一般的搜索结果来说，得到最后的排序结果以后就可以了。而对于广告来说，我们还需要对点击率进行准确的预测。这个我们之前提到过。于是在这篇文章中专门提到了如何从配对结果到点击率的预测。
			- 配对法学习排序完成以后的广告之间顺序是绝对的，但是绝对的数值可能是不太精确的。这里进行校准的目的是根据配对法产生的预测值，再去尽可能准确地转换为实际的点击率的数值。一般来说，这里就可以再使用一次对数几率回归。也就是说，这个模型的唯一特性就是配对法产生的预测数值，然后模型的目的是去估计或者说是预测最后的实际数值。这种使用一个回归模型来进行校准的方法，也用在比如要将支持向量机的结果转换成概率结果这一应用上。
			- 原理上讲，先有一个配对模型进行排序，然后再有一个校准模型对模型的绝对估计值进行重新校正，这是很自然的。但是在实际的工业级应用中，这意味着需要训练两个模型，那无疑就变成了比较繁复的步骤。所以，在这篇文章里，作者们想到了一种结合的办法，那就是结合单点法和配对法。
				- 具体来说，就是直接把两者的目标函数串联在一起。这样做的好处是，可以直接用现在已经有的训练方法，而且同时解决了排序和更加准确预测点击率的问题。
				- 串联多个目标函数是经常使用的一种技术。其目的和作用也就和这个串联的想法一样，就是希望针对多个不同的目标进行优化。一般来说，这里面的核心是，多个串联的目标函数需要共享模型参数才能形成有关联的总的大的目标函数；如果没有共享参数，那就仅仅是一种形式上的串联。
		- 前分享的 Facebook 的解决方案，并没有真正考虑往信息流里插入广告的难点，也就是广告的排序，依然把广告的排序问题当做分类问题，也就是用对数几率回归（Logistic Regression）来解决。
			- 这篇文章里，作者们也是用了 Facebook 提出的“归一化的交叉熵”，简称 NE 的概念以及业界比较常用的 AUC 来针对模型进行线下评价。
- 阿里巴巴广告预估
	- 从广告点击率预估的大规模数据中学习多段线性模型》（Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction）[1]这篇文章中，作者们提出了一种多段线性模型来解决我们刚刚说的这两个问题，这个模型简称为 LS-PLM（ Large Scale Piecewise Linear Model ）。
		- 第一，就是数据中呈现的非线性化的关系。也就是说，我们的模型必须在某一个地方考虑到特性之间的非线性表征，以及对于目标标签的非线性关系。
		* 第二，就是数据的不均衡以及数据的稀疏性。有很多广告商是新广告商，很多广告是新广告。在这样的情况下，我们就必须要处理“冷启动”和“热启动”这两种局面。
	- 既然数据在整个空间里可能呈现非线性的关系，那么我们是否能够把整个空间分割成较小的区域，使得每个区域内依然可以使用线性模型来逼近这个区域内的数据点呢？其实在统计学习中，这种模型常常被叫作“混合模型”。在很多机器学习教科书中都会讲授的一种混合模型是“高斯混合模型”（Gaussian Mixture Model）。
		- LS-PLM 在这篇论文的实际应用中，基本上可以被理解成为一种混合线性模型。这个模型的一个子模型叫作“分割函数”，也就是模型需要学习每一个数据点到底是依赖于哪一个线性模型来进行预测的。当然，这个分割是一种概率的分割。实际上，每一个数据点都依赖所有的线性模型来进行预测，只不过对每个模型的依赖程度不一样。对于每一个不同的线性模型来说，最大的不同就是每一个模型有自己的系数。也就是说，之前只有一个全局模型并且只有一组系数，相比之下，这里有多组系数来决定模型的预测效果。
		* 很明显，对于 LS-PLM 来说，每一个局部都是线性的，但是在整体上依然是一个非线性的模型。LS-PLM 还借助了两种正则化机制。一种叫作 L1 正则，这种正则化主要是希望模型保留尽可能少的特性，从而达到对于模型特性的选择。另外，模型还采用了一种 L2,1 正则的方法，这种方法的目的也是特性选择，但是希望能够把一组特性全部选择或者全部置零。
		- 作者们尝试了不同数目的数据分割，从 2 个到 36 个不等。最终，他们发现当数据分割为 12 个的时候，模型的效果达到最优，而之后，模型效果并没有明显提升。最终推出模型的 AUC 比直接使用一个对数概率回归的全局模型，效果要好 1.4%。
	- 1. Kun Gai, Xiaoqiang Zhu, Han Li, Kai Liu, Zhe Wang. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction. CoRR abs/1704.05194 , 2017.2. 
	* Tiezheng Ge, Liqin Zhao, Guorui Zhou, Keyu Chen, Shuying Liu, Huiming Yi, Zelin Hu, Bochao Liu, Peng Sun, Haoyu Liu, Pengtao Yi, Sui Huang, Zhiqiang Zhang, Xiaoqiang Zhu, Yu Zhang, Kun Gai. Image Matters: Jointly Train Advertising CTR Model with Image Representation of Ad and User Behavior. CoRR abs/1711.06505 , 2017.3. 
		- 广告点击率预估和图像处理的结合
		- 这篇文章结合了近期好几个利用深度学习来进行图像处理和广告点击率预估的工作。首先，就是所有的特性都利用一个“嵌入层”（Embedding Layer）把原始的特性转换成为数值特性。这种思路我们在之前介绍文本处理，特别是 Word2Vec 的时候曾经进行了详细的讲解。
		* 而在这里，不管是文本信息还是图像信息，都根据自己的特点转换成为了数值特性。这里我们要解决的一个核心问题，就是用户和广告之间的匹配问题，这篇论文的模型是这么处理的。首先，对所有广告的 ID 及其图像进行单独的嵌入。然后对用户过去的喜好，特别是对图像的喜好进行了另外的嵌入，然后这些嵌入向量形成用户的某种“画像”。
		* 用户的画像和广告信息的嵌入被直接串联起来，形成最终的特征向量。在此之上，利用一个多层的神经网络来学习最后的点击率的可能性。在深度学习建模中，这种把多种来源不同的信息通过简单的拼接，然后利用多层神经网络来进行学习的方法非常普遍和实用。在这篇论文的介绍中，除了在模型上对图像进行处理以外，还有一个创新，就是提出了一个叫“高级模型服务器”（Advanced Model Server），简称 AMS 的架构理念。
		* AMS 是针对深度学习模型的大计算量而专门打造的计算体系。总体来说，AMS 的目的是把深度学习模型中的很多基础步骤进行拆分，然后把这些步骤部署到不同的服务器上，从而能够把复杂的模型拆分成细小的可以互相交流的步骤。从最终的实验结果上来看，基于深度学习的模型要比对数几率回归的模型好 2~3%。
	* Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Xiao Ma, Yanghui Yan, Xingya Dai, Han Zhu, Junqi Jin, Han Li, Kun Gai. Deep Interest Network for Click-Through Rate Prediction. CoRR abs/1706.06978 , 2017.
		- DIN 依靠一种基本的模型架构，那就是先把所有的特性变换成嵌入向量，然后针对不同的特性进行划组，一些特性得以直接进入下一轮，另一些特性经过类似图像中的池化（Pooling）操作抽取到更加高级的特性。之后，所有的特性都被简单串联起来，然后再经过多层的深度神经网络的操作。
		* DIN 在这个架构的基础上，提出了一种新的“激活函数”（Activation Function），叫 DICE，目的是可以在不同的用户数据中灵活选择究竟更依赖于哪一部分数据。可以说，在某种意义上，这个架构非常类似深度学习中比较火热的 Attention 架构，其目的也是要看究竟那部分数据对于最终的预测更有效果。
		* 从最后的实验中看，不管是在内部数据还是外部公开的例如 MovieLens 或者 Amazon 的数据上，基于 DIN 的模型都比线性模型和其他的深度学习模型有显著的提高。
* 拍卖机制-诺贝尔奖论文
	- Jun Wang, Weinan Zhang and Shuai Yuan. Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting. Foundations and Trends® in Information Retrieval: Vol. 11: No. 4-5, pp 297-435, 2017.
- 竞价Bidding
	- Wu, W. C.-H., Yeh, M.-Y., and Chen, M.-S. Predicting winning price in real time bidding with censored data. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1305–1314. ACM, 2015.
- Optimal Bidding
	- Ramakrishna Gummadi, Peter Key and Alexandre Proutiere. Repeated auctions under budget constraints: Optimal bidding strategies and equilibria. In Eighth Workshop on Ad Auctions, 2012.
	* Yuan, S., Wang, J., and Zhao, X. Real-time bidding for online advertising: measurement and analysis. Proceedings of the Seventh International Workshop on Data Mining for Online Advertising, page 3. ACM, 2013. 
	* Andrei Broder, Evgeniy Gabrilovich, Vanja Josifovski, George Mavromatis, and Alex Smola. Bid generation for advanced match in sponsored search. Proceedings of the fourth ACM international conference on Web search and data mining (WSDM '11). ACM, New York, NY, USA, 515-524, 2011.
- bidding optimization
	* [1]Perlich, C., Dalessandro, B., Hook, R., Stitelman, O., Raeder, T., and Provost, F. Bid Optimizing And Inventory Scoring in Targeted Online Advertising. Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 804–812. ACM, 2012.2. 
		- 线性出价策略在实际操作中比较方便灵活，在这篇论文中，这种算法也取得了比较好的效果。不过遗憾的是，这种做法并没有太多的理论支持。
	* Zhang, W., Yuan, S., and Wang, J. Optimal Real-Time Bidding for Display Advertising. Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 1077–1086. ACM, 2014.3. 
	* Zhang, W., Ren, K., and Wang, J. Optimal Real-time Bidding Frameworks Discussion. arXiv preprint arXiv:1602.01007, 2016.4. 
	* Zhang, W. and Wang, J. Statistical Arbitrage Mining for Display Advertising. Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1465–1474. ACM, 2015.
- pacing， budget pacing
	- 1. Lee, K.-C., Jalali, A., and Dasdan, A. Real Time Bid Optimization with Smooth Budget Delivery in Online Advertising. Proceedings of the Seventh International Workshop on Data Mining for Online Advertising, page 1. ACM, 2013.2. 
	* Xu, J., Lee, K.-c., Li, W., Qi, H., and Lu, Q. Smart Pacing for Effective Online Ad Campaign Optimization. Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 2217–2226. ACM, 2015.3. Agarwal, D., Ghosh, S., Wei, K., and You, S. Budget Pacing for Targeted Online Advertisements at Linkedin. 
	* Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1613–1619. ACM, 2014.4. 
	* Chen, Y., Berkhin, P., Anderson, B., and Devanur, N. R. Real-time Bidding Algorithms for Performance-based Display Ad Allocation. Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1307–1315. ACM, 2011.5. Zhang, W., Rong, Y., Wang, J.,


## HW

* Newton's Method and L-BFGS method
	- 推导 论文
	- https://www.hankcs.com/ml/l-bfgs.html
- Model Evaluation - AUC

字节的广告模型
https://arxiv.org/abs/1909.03602
https://zhuanlan.zhihu.com/p/133140002

probit, logit, tobit model
二价竞拍 究竟是为啥？

## 场景
穿山甲是什么角色？DSP + DMP

第一步，用户来到某个网站，网站产生了一个对实时竞价系统广告网络的请求。
第二步，实时竞价系统广告网络向某个 DSP 发送请求，这个请求里包含了用户是谁，当前页面是什么，以及一些上下文的数据。
第三步，DSP 收到请求以后，就向 DMP 发送一个数据请求，用于收集用户更多的信息，包括用户的年龄、性别以及喜好。
第四步，DSP 收到 DMP 的信息以后，会向实时竞价系统发出一个自己认为合适的广告以及竞价的价格。
第五步，实时竞价系统广告网络收集到所有的广告竞价以后，会举行一个拍卖（Auction）。
每个实时竞价系统的拍卖规则可以不同。
第六步，实时竞价系统会向赢得广告位的 DSP 发送最后的收款价格，这个价格是根据某种拍卖规则决定的。
第七步，广告显示给了用户。第八步，用户对广告的反馈，例如是否点击，是否购买广告相应的产品，是否订阅广告对应的服务等，这些信息会返回给 DSP。

## References

* Google AdWords
- wiki-online ads
	- https://en.wikipedia.org/wiki/Online_advertising
- computing ads
	- HotWired AT&T, Sponsored Search, AdWords -> 2005 RTB
- Stanford Course
- google财报 fb财报 战略报告