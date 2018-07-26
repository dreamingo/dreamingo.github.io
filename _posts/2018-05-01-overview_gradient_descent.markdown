---
layout: post
title: "梯度下降优化方法进化史与并行概况"
modified:
categories: Machine Learning
description: "Residual Network详解"
tags: [graident-descent optimization]
comments: true
mathjax: true
share:
date: 2018-05-01T11:04:12+08:00
---

本文主要结合 Reference 中的资料（主要是[An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)）对各种梯度下降优化进行阐述和总结，并辅以一些个人的小见解。

## Reference
* [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)
* [知乎专栏：Adam那么棒，为什么还对SGD念念不忘 (1) —— 一个框架看懂优化算法](https://zhuanlan.zhihu.com/p/32230623)
* [知乎专栏：比Momentum更快：揭开Nesterov Accelerated Gradient的真面目](https://zhuanlan.zhihu.com/p/22810533)
* [Hogwild!: A Lock-Free Approach to Parallelizing](https://arxiv.org/pdf/1106.5730.pdf)
* [Parallel Machine Learning with Hogwild!](https://medium.com/@krishna_srd/parallel-machine-learning-with-hogwild-f945ad7e48a4)
* [Downpour SGD(大规模分布式深度网络)](https://wlypku.github.io/2016/10/06/Downpour-SGD/)

## 梯度下降的变种

对于梯度下降方法，其中共有三个变种。这些变种之间的区别，主要都是集中于我们使用**多少的数据量**来计算目标函数的梯度。这主要涉及到在更新过程中，
对精度和训练时间的一个trade-off。

* **Batch gradient descent**: 每一次更新，就要遍历**所有**的训练数据来计算损失函数的梯度。
对于凸函数，batch gd 能够保证下降到全局最小点处，而对于非凸函数，也能保证下降到局部最优解中。

* **Stochastic gradient descent**：每次一更新，随机选择一个训练样本来计算损失函数的梯度，并用该梯度来更新参数。
因为每次只用一个训练样本来更新，因此速度会快得多，并且能够支持在线学习的需求（online-learning）。但是SGD仅仅使用单个样本进行更新，会引入高方差效应(high variance)，从而导致目标函数在训练过程中会产生剧烈的震荡。对于这个震荡现象，我们可以从两方面来看：

    * SGD的震荡性一定概率上允许在训练过程中，跳出局部最小值的困境，去寻找一个更好的最优解；
    * 这个震荡性有可能令其难以收敛。但是实际应用证明，通过缓慢的降低学习速率，SGD能够与Batch Gradient descent有大致的收敛行为。

* **Mini-batch gradient descent** 每次计算一个batch(通常是64-256左右)样本的梯度来更新参数。综合了前两者的优势，既保证了稳定的收敛行为，减少内存使用和时间消耗。同时，也能利用各大机器学习框架中深度优化的矩阵计算框架来加速计算过程。在很多实际应用中，我们指代的SGD就是Mini-batch gradient descent。


对于上面三种基本的梯度下降方法，在实际应用过程中会有许多的困难与挑战，主要归纳为以下几方面：

* 难以指定合适的学习速率，同时对于学习速率的调整策略(例如指定最低阈值、每隔多少个epho后lr进行衰减)，也需要各种先验人工知识。
* 不同的参数更新频繁程度不一样(在稀疏问题的求解上很常见)，这就导致了我们希望针对不同的参数来指定不同的学习速率。
* 近年来有文献指出，高度非凸问题的优化困难，不在于局部最优解，而主要来源于SGD在**鞍点(saddle point)**时无法逃脱的问题。所谓的鞍点(可以参考[【最优化】鞍点介绍](https://zhuanlan.zhihu.com/p/33340316))，是指一个不是局部最小点的驻点（一阶导数为0），从该点出发的一些方向是函数的极大值点，而另外一些方向则是极小值点。因为鞍点的导数几乎为0，并且其二阶的海森矩阵是不定的，这导致了SGD算法难以逃脱它的魔掌。


<figure>
	<a href=""><img src="/assets/images/gd/saddle_point.png" alt="" width="300" heigh="300"></a>
	<figcaption><a href="" title="saddle_point">函数$z = x^2 - y^2$在(0, 0)点的鞍点</a>.</figcaption>
</figure>

## 梯度下降的优化方法

在接下来的文章中，我们将会着重介绍优化算法从SGD -> SGDM -> NAG -> AdaGrad -> AdaDelta -> Adam -> NAdam 的发展历程。其中
在参考文章中[Adam那么棒，为什么还对SGD念念不忘 (1) —— 一个框架看懂优化算法](https://zhuanlan.zhihu.com/p/32230623)中提出了一个利用
**一阶动量**和**二阶动量**来归纳上述优化算法的框架。在这里，我先简单介绍一下上面提到的两个新名词：

* **历史梯度的一阶动量**: 指代历史梯度向量的累加值，因为向量值，所以正负，有方向。着重点在**方向的调控**。
* **历史梯度的二阶动量**: 指代历史梯度的平方和累加量，着重点在历史梯度大小的累积量。

### SGD with Momentum(SGDM)

前面提到，SGD一个主要的问题是容易引起震荡。因为Momentum的引入是为了抑制SGD的震荡行为。是在SGD的基础上引入了一阶动量：

$$
\begin{equation}
\begin{aligned}

v_t &= \gamma v_{t-1} + \eta\nabla_{\theta}J(\theta) \\
\theta &= \theta - v_t

\end{aligned}
\end{equation}
$$

在上面的式子中，$$v_t$$记录的是历史的梯度方向的移动平均值，也就是之前提到的**历史梯度的一阶动量**。而参数$$\gamma$$的经验值一般是0.9。

$$t$$时刻的下降方向，是由历史梯方向和当前的梯度方向一起决定的。这样可以一定程度上避免当前梯度方向变化过大而引起的震荡行为。同时我们可以看到，如果在某一子方向$$i$$上，历史梯度值$$v^{t-1}_i$$为正，并且当前梯度的该子方向值$$\nabla_{\theta}J(\theta)_i$$也为正，则起到了一个加速的作用。反之则起到了一个抑制、减速的效果。因此SGDM与SGD比起来，其收敛速度会更快。

### Nesterov accelerated gradient(NAG)

在上面SGDM的式子(1)中，我们作一些简单的变化。$$\theta = \theta -  \gamma v_{t-1} - \eta\nabla_{\theta}J(\theta)$$。可以看到的是每一更新中，$$\gamma v_{t-1}$$这个量肯定是要走的。那么我们为啥不先偷走这一步，然后再根据那里的梯度再去探索前进呢？这就有点『偷窥未来』的意味了。


$$
\begin{equation}
\begin{aligned}

v_t &= \gamma v_{t-1} + \eta\nabla_{\theta}J(\theta - \gamma v_{t-1}) \\
\theta &= \theta - v_t

\end{aligned}
\end{equation}
$$

从上式我们可以看到，我们先让参数$$\theta$$走到$$\theta - \gamma v_{t-1}$$的地方。然后再延续前面SGDM的讨论来进行更新。这种偷走一步的方法，令SGD能够有可能走出局部最优解的困境。想想当你被困在一个山谷中，四周都是略高的山脉，此时你觉得没有可以下坡的地方了。但是如果你先偷走一步，爬上小山坡，会发现外面的世界还很辽阔。

除了协助SGD走出局部最优解外，这种『偷窥未来』的方式还可以协助SGD跑的更快更稳，试想一下因为偷走的这一步，我们发现的未来的地势更陡/更平坦，那么就会改变我们的行走策略了。例如如果前面的梯度比当前位置的梯度大，那我就可以把步子迈得比原来大一些，如果前面的梯度比现在的梯度小，那我就可以把步子迈得小一些。这个大一些、小一些。**总体而言，NAG的收敛速度比SGDM更快。**

从数理的角度来看，参考文章[『比Momentum更快：揭开Nesterov Accelerated Gradient的真面目』](https://zhuanlan.zhihu.com/p/22810533)中，通过公式的变化，发现其实在NAG算法中：

> **本质上是多考虑了目标函数的二阶导信息，怪不得可以加速收敛了！其实所谓“往前看”的说法，在牛顿法这样的二阶方法中也是经常提到的，比喻起来是说“往前看”，数学本质上则是利用了目标函数的二阶导信息。**


<figure>
	<a href=""><img src="/assets/images/gd/nag.png" alt="" width="650" heigh="300"></a>
	<figcaption><a href="" title="saddle_point">左图是SGDM的优化轨迹，右图是NAG的优化轨迹</a>.</figcaption>
</figure>

- - -

上面的介绍的 SGDM 和 NAG 算法，通过加入一阶动量的方式，在解决 SGD 易于震荡，难以收敛的问题的同时，加快了训练过程中收敛的速度。
但是前面提到**学习速率调整难**的问题还没有得到有效的解决。因此，下面介绍的几种方法，主要是针对学习速率的调整。

### Adagrad

Adagrad 优化算法的主要贡献是：在不同的时刻，根据不同的参数，给予不同的学习速率。
**对于频繁更新的参数，我们希望其学习速率越来越小；而对于更新很少的参数，我们则希望每次更新是能够尽可能的激进一些。**
正是这个原因，这个算法本身非常适合稀疏数据的求解。例如：

* 在 CTR 问题中，输入数据往往是 OneHotEnocde 后的01向量，而1出现往往会集中在某几个特征中。因此参数中某些值就会被更新的很频繁。
* 在 WordEmbeddings 在，某些词语出现频次很低，因此这些词语需要更大的更新（对比于那些频繁的单词）。

对于每个参数$$\theta_i$$，我们先定义该参数在时刻$$t$$的『二阶动量』（历史梯度的平方和的累积量）：$$G_{t,i} = \sum_{t'= 1}^t (\nabla_{\theta_{t'}}J(\theta_{t', i}))^2$$
其在一定程度上反应了参数被更新的频繁程度。

$$
\begin{equation}
\begin{aligned}

\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, i} + \epsilon}} . \nabla_{\theta_t}J(\theta_{t, i}) \\
\end{aligned}
\end{equation}
$$

上面的$$\epsilon$$算是一个平滑项，主要是防止分母为0的情况。Adagrad一定程度上解决了学习速率自适应和不同参数不用学习速率的两个问题。但是其缺点也很明显：**这种单调下降的学习速率自适应的方法有些过于激进，因为历史累计的梯度平方值一直在变大，这导致了学习速率在后期的时候会变得非常的小，导致训练提前结束。**

### Adadelta

Adadelta是Adagrad的一种改进方法，与Adagrad激进的方法相比，其不会累积所有历史的梯度平方值，而是设置一个窗口的大小，只累积过去一段时间内的梯度值。

然而，在实际的算法中，却没有通过低效的存储$$w$$个历史梯度平方值，而是通过一种**平均累积**的方法来计算梯度的『二阶动量』：

$$
\begin{equation}
\begin{aligned}

G_{t, i} = \gamma G_{t-1, i}  + (1 - \gamma)g_t^2

\end{aligned}
\end{equation}
$$

最后其更新方式，与Adagrad一致。

$$
\begin{equation}
\begin{aligned}

\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, i} + \epsilon}} . \nabla_{\theta_t}J(\theta_{t, i}) \\
\end{aligned}
\end{equation}
$$

- - -

谈到了这里，Adam的出现就呼之欲出了，为什么呢？我们在前面提到：
* 利用一阶动量的SGDM和NAG，解决了SGD震荡和训练缓慢的问题
* 利用二阶动量的Adagrad和Adadelta，解决了学习速率自适应的问题。

那为什么不将这两者结合起来呢？于是就有了Adam算法(Addative and momentum)：

### Adam

在这里，我们先定义历史梯度的一阶动量和二阶动量：


$$
\begin{equation}
\begin{aligned}

m_t &= \beta_1 m_{t-1} + (1 - \beta_1)g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_v)g_t^2
\end{aligned}
\end{equation}
$$

其中参数$$\beta_1, \beta_2$$分别就是这个算法的两个超参了，分别控制一阶动量和二阶动量。

而参数的更新方式，则是：

$$
\begin{equation}
\begin{aligned}

\theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{v_t} + \epsilon}m_t
\end{aligned}
\end{equation}
$$

### NAdam (Nesterov Adam)

前面谈到的Nesterov方法中『偷窥未来』的思想那么超前，那干嘛不加到Adam中呢？于是就有了 NAdam 算法：

通过上式中7中，我们知道$$\beta_1.m_{t-1} / \sqrt{V_{t-1} + \epsilon}$$这个量是必须要走的。因此。根据NAG的算法，我们计算的梯度变成了：

$$
g_t = \nabla_{\theta_t}f(\theta_t - \beta_1.m_{t-1} / \sqrt{V_t})
$$

可以看出，Nadam对学习率有了更强的约束，同时对梯度的更新也有更直接的影响。一般而言，在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。

## 如何选择合适的优化算法

在引用文章[『Adam那么棒，为什么还对SGD念念不忘 (2)—— Adam的两宗罪』](https://zhuanlan.zhihu.com/p/32262540)，[『Adam那么棒，为什么还对SGD念念不忘 (3)—— 优化算法的选择与使用策略』](https://zhuanlan.zhihu.com/p/32338983)中，分别讨论的Adam算法的一些问题，以及何时该选择哪一种算法的方案。这里珠玉在前，我就不献丑了。

## SGD算法的并行

在最前面我们介绍SGD算法时，我们都知道该算法本质上是一个序列执行的算法：『每次随机选择一个样本更新参数；然后再选择一个样本...』。因此如何有效的并行SGD算法是业界中一直持之以恒研究的问题。

### Hogwild!

Hogwild!是一个异步 Lock-free 版本的SGD改良算法(Asynchronous SGD)。该算法主要致力于利用多核CPU的并行能力来加快SGD算法的运行过程。这个算法本质非常的简单粗暴：

> 程序将待更新的参数放置在共享内存(shared-memory)中，每个线程/进程随机选择一个/Batch的样本对参数进行更新。该过程是一个无锁并行算法。

可以看到的是，这个算法强行把原来序列执行的SGD变成了多线程并行执行的算法。这种 Lock-free 的算法会有可能有『冲突』：
1. 多个线程同时更新，涉及到**复写**同一个参数；
2. 每个线程在更新的时候，参数有可能已经被其他线程更新过了，不是最新的数据。

因此Hogwild!的核心思想在于当算法发生冲突时还有效吗？答案是Yes！当然详细的数学证明过程可以参考论文。这里仅仅从一些个人方面来进行讨论：

* SGD更新过程是随机抽取样本进行更新参数的。随机的情况下导致参数冲突的概率就会降低。特别是当**数据稀疏**的情况下更是如此。因此Hogwild!非常适用于稀疏问题的并行求解上。
* 即使是发生了冲突，也并非完全是按照坏的方向去发展。毕竟大家都还是朝着梯度下降的方向去走。可能只是有稍微的偏差。
* 民间中有一个既是笑话，也是真理的名言：**SGD is so robust that an implementation bug is essentially a regularizer**，每次冲突的发生导致微小的偏离，我们可以将其理解为一个正则项。

### Downpour SGD

> Downpour一词翻译为『倾盆大雨』，在这个算法中重要意味着并行的更新各个参数，就像大雨一般，雨点从多点落下。

Downpour SGD是一个应用于Google DistBelief（TensorFlow前身）异步的SGD算法。与前面的Hogwild!算法不一样。该算法致力于将SGD任务切分成若干并行的任务运行在不同的机器上，
通过框架中通信协议进行参数、计算结果之间的同步与汇合。


<figure>
	<a href=""><img src="/assets/images/gd/downpour.png" alt="" width="400" heigh="400"></a>
</figure>

从上图中我们可以大概的看出其基本方法：将训练集划分称为若干子集，并对每个子集运行一个单独的模型副本。模型副本之间的通信均通过中心的参数服务器(parameter-server)，
该中心服务器维护了模型参数的单独状态，并分割到多台机器上。算法中的异步，主要体现在：

1. 每个模型副本之间是独立运行的
2. 参数服务器中各个节点之间同样是独立的。

考虑Downpour SGD的一个最简单的实现，在处理每个mini-batch之前，模型副本都会向参数服务器请求最新的模型参数。因为DistBelief框架也是分布在多台机器上，
所以其框架的每个节点只需和参数服务器组中包含和该节点有关的模型参数的那部分节点进行通信。在DistBelief副本获得更新后的模型参数后，
运行一次mini-batch样本来计算参数的梯度，并推送到参数服务器，以用于更新当前的模型参数值。

与同步算法相比，如果一台机器失效了，则所有可能会用到该参数的其他机器都会面临着延时的风险。而异步算法在一台机器失效时，其他模型副本依然会我行我素的更新自己的参数并推送到服务器上。
这就会面临着与Hogwild!中**参数过时**的问题。这种缺乏安全操作理论的方法看似比较危险，但是实际上由于**引入新的随机性和SGD算法本身的鲁棒性足够的强，从而发现这种放松一致性的做法，
在获得并行加速的同时丝毫不会影响算法精度。**

### TensorFlow

> 具体可以参考[Distributed Tensorflow Doc](Distributed Tensorflow Doc)

TensorFlow中主要吸收了DistBelief中的经验。其本质上便是通过将一个大的计算过程通过划分为独立子任务，并通过中控服务器进行同步与更新的并行过程。
TensorFlow中通过将计算图划分为独立的子图并分发到各个机器独立计算，同时通过通信来更新和同步参数。
