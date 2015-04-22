---
layout: post
title: "Standford公开课《机器学习》笔记 - Linear regression"
modified:
categories: 机器学习与数据挖掘
description:
tags: [Standford, Machine Learning, Linear regression]
image:
    feature: abstract-5.jpg
    credit:
    creditlink:
comments: true
mathjax: false
share:
---


重新又看了一次Andrew Ng的机器学习公开课的视频，决定每一课都写点笔记，加深印象。**仅限于对个别重点的个人笔记，因此一些简单的细节会有所忽略**。

### [基本定义]

第一课中的房屋价格预测问题， 首先它是一个有监督学习的问题（对于每个样本的输入，都有正确的输出或者答案），同时它也是一个回归问题（预测一个实值输出）。

训练集表示如下：

![example]({{site.url}}/images/linear_regression/examples.png)

其中，在这里先定义若干符号：

* $n$: 样本特征的数量（在例子中，n=2,分别是房屋的大小和bedroom的个数）
* $m$: 训练样本的数目（在上例中，假设有100个训练样本，则m = 100）
* $$x^{i}_{2}$$: 第i个sample中的第2个特征。（x的上标$$x^{i}$$表示第i个训练样本, x的下标表示x的第几个特征。$$x^{i}_{j}$$）
* $y^{i}$: 第i个sample的结果y值。

<!--more-->

在监督学习中，经过trainning set的训练后，hypotheses函数对新的例子进行预测，并输出结果。在上述例子中，我们定义了如下的hypotheses函数$h_{\theta}(x)$

$$ h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} $$

为了方便表示，令$x_{0} = 1$, 于是可以简化上述等式：

$$ h_{\theta}(x) = \sum_{i=0}^{n}\theta_{i}x_{i} = \theta^{T}x $$

同时定义Cost Function$J(\theta)$如下：

$$ J(\theta) = \frac{1}{2}\sum_{i=1}^{m}{(h_{\theta}(x^{(i)}) - y^{(i)})}^2 $$

### [Gradient descent]

我们企图通过选择$\theta$来最小化$J(\theta)$.而Gradient descent的思路是先初始化一个$\theta$, 
然后每次选择$J(\theta)$最**陡峭**的方向（函数的梯度方向）下降，直到达到最低点为止。可是值得注意的一点是，当损失函数不是凸函数的时候，
**根据初始参数的不同，梯度下降方法有可能达到局部最优点，不一定达到全局最优点。**


<img src="{{site.url}}/images/linear_regression/gd1.png" width="300px" />
<img src="{{site.url}}/images/linear_regression/gd2.png" width="300px" />

在做Gradient descent的时候，一般会进行`feature scaling`,即是将各个特征标准化，会有效加快梯度下降的速度，使得各个特征的下降速度不会相差过远。使其取值范围在$-1<= x <= 1$, 一般会使

$$x = \frac{x - \overline{x}}{max - min}$$


梯度下降算法，重复下面公式直至收敛。其中$\alpha$表示学习率，如果$\alpha$过小，梯度下降可能很慢；如果过大，梯度下降有可能“迈过”（overshoot）最小点，并且有可能收敛失败，并且产生“分歧”( diverge )

$$ \theta_{j} := \theta_{j} - \alpha\frac{\partial}{\partial\theta_{j}}J(\theta) $$

而其中，根据$h_\theta(x), \frac{\partial}{\partial\theta_{j}}J(\theta)$的推导如下：

<img src="{{site.url}}/images/linear_regression/partial.png" width="500px" />

因此，$\theta$的更新公式为：$$ \theta_{j} := \theta_{j} - \alpha\sum_{i}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)} $$

上面提到的，称为**`batch gradient descent`**,而一般在大规模机器学习中，更常用的则是**`SGD（stochastic gradient descent）`**，其算法如下：

<img src="{{site.url}}/images/linear_regression/sgd.png" width="500px" />

`batch gradient descent`[`GBD`]每次都必须使用所有的样本来更新每一个$\theta$，而`SGD`在一轮的迭代中，对比与`GBD`, 主要是少了$\sum$符号，每次只用一个训练样本更新所有的$\theta$。因此，在训练数据很大的时候，`SGD`速度上明显更加优先。（虽然`SGD`不是每次迭代得到的损失函数都向着全局最优方向， 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近。`SGD`通常`比batch gradient descent`更快**接近**$\theta$的最小值，要注意SGD也许永远都**不会收敛**于最小值，并且`SGD`中得到的$\theta$会一直徘徊在$J(\theta)$最小值附近，但是在实际应用上，接近最小值的效果对比与最小值的效果，都已经足够的优秀了）

### [Normal Equation]

如果损失函数是二次的话，那么这时候我们可以**矩阵求导**的方法快速求解出损失函数$J(\theta)$的最小值。具体推导过程如下：

<img src="{{site.url}}/images/linear_regression/ne1.png" width="500px" />
<br>
<img src="{{site.url}}/images/linear_regression/ne2.png" width="500px" />

上述的推断设计到大量的线性代数知识，有兴趣的同学可以参考这门课的lecture note1来进行推导。最终，我们可以求出$\theta = (X^{T}X)^{-1}X^{T}\overrightarrow{y}$

从上面可以看出，`normal equation`中需要对矩阵进行**求逆**，当矩阵过大的时候，时间和空间复杂度都是巨大的。因此，当X矩阵规模较小的时候，使用`normal equation`来进行求解
不失为一个好办法:简单，方便，避免了feature scaling。

### [Probabilistic interpretation]

在课程中，Andrew Ng提出了为什么当我们面对一个回归问题的时候，要选择用linear regression，为什么损失函数又选择最小二乘法？最后，他提出了通过提出最大似然估计方法，解决了上述问题。

在这里，我们先假设$ y^{(i)} = \theta^{T}x^{(i)} + \epsilon^{(i)}$, 在这里，$\epsilon^{(i)}$代表了随即误差（例如一下未建模特征/random noise），在这里我们假设误差
$\epsilon$满足正态分布（mean为0, variance 为$\sigma^{2}$ ）且IID（independent and identically distributed）

$$ p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(\epsilon^{(i)})^{2}}{2\sigma^{2}})$$

这意味着：

<img src="{{site.url}}/images/linear_regression/likehood.png" width="400px" />

同时我们定义likelihood函数$L(\theta) = L(\theta;X, \overrightarrow{y}) = p(\overrightarrow{y}\mid{X;\theta})$..[似然函数可以理解为给定参数$\theta$和X，y出现的概率]。
因此：

<img src="{{site.url}}/images/linear_regression/likehood2.png" width="500px" />

而所谓的**最大似然[maximum likelihood],则是意味着通过选择参数$\theta$，使得数据出现的概率尽可能的大**,而往往为了使得似然函数最大化，将其取log化是一般优化的方法之一;

<img src="{{site.url}}/images/linear_regression/likehood3.png" width="500px" />

从上图可以看出，要使得$\ell(\theta)$最大，即是要等式的后面部分$\sum_{i=1}^{m}(y^{(i)} - \theta^{T}x^{(i)})^{2}$尽可能的小。因此可以看出，这种基于最小二乘法（least-squares）的回归方法是对应于对$\theta$的最大似然估计上的。

###[参考资料]

[Coursera公开课笔记](http://52opencourse.com/83/coursera%E5%85%AC%E5%BC%80%E8%AF%BE%E7%AC%94%E8%AE%B0-%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AC%AC%E4%BA%8C%E8%AF%BE-%E5%8D%95%E5%8F%98%E9%87%8F%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92-linear-regression-with-one-variable)

[Standford-C229 lecture notes1](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
