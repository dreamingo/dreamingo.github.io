---
layout: post
title: "Residual Network详解"
modified:
categories: DeepLearning
description: "Residual Network详解"
tags: [cnn deeplearning residual]
comments: true
mathjax: true
share:
date: 2018-02-01T11:24:30+08:00
---

本文主要结合 Reference 中的资料，介绍、总结 Residual Network 出现的缘由、网络的原理以及相关细节的梳理；

## Reference
* Residual Network(ResNet)
    * [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    * [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027v3.pdf)
* [2016_tutorial_deep_residual_networks](2016_tutorial_deep_residual_networks)
* [知乎专栏：给妹纸的深度学习教学(4)——同Residual玩耍](https://zhuanlan.zhihu.com/p/28413039)

## 背景介绍

在论文的最开端，何凯明他们就提出了一个问题： _Is learning better networks as easy as stacking more layers?_ 在论文中，
他们指出了在 plain-network 上直接无脑的堆砌，会出现一种 degradation(退化) 的现象。也就是说更深的网络，反而有更深的训练误差；

<figure>
	<a href=""><img src="/assets/images/residual/degradation.png" alt="" width="500" heigh="300"></a>
</figure>

针对上面的问题，作者又提出了一种新的深层网络的构造方式：在一个浅层网络的基础上，添加更多的层，已有的层则用浅层网络的参数
进行初始化，新添加的层作为一个『identity mapping』，即$$f(x) = x$$。按照这种假设，那么这个深层网络不应该会比浅层
网络有更高的 trainning-error。但是实际实验证明，这种构造出来的深层网络并不能很好的得到优化(Optimization solver cannot find the solution when going deeper)。
基于上面已有的这些问题，论文提出了 Deep Residual Learning的架构。

## Residual Block
我们首先定义$$H(x)$$是若干个堆砌的卷积层所指代的函数mapping。在这里，$$x$$表示这些层的输入。在已知的理论中，我们知道
少量几个堆叠的非线性神经网络就可以近似的模拟任何的复杂函数。那么我们也可以**认为它可以近似的模拟残差函数**(residual-functions): $$F(x) = H(x) - x$$
因此，在这个函数中，我们的学习目标从$$H(x)$$变成了残差函数$$F(x)$$。
<figure>
	<a href=""><img src="/assets/images/residual/residual_block.png" alt="" width="600" heigh="400"></a>
</figure>

作者指出，残差函数$$F(x)$$比起原来的映射$$H(x)$$更容易优化。为什么呢？因为**当最优解的函数近似于一个identity-mapping的时候（请看下图），
那么$$F(x)$$函数对于这些微小的变化更容易得到优化。残差的引入去掉了主体部分，从而突出了微小变化**

<figure>
	<a href=""><img src="/assets/images/residual/optimize.png" alt="" width="600" heigh="400"></a>
</figure>

## 网络结构与细节

在这里，我们仅仅截取了34-layer residual network 的部分结构图。其中有部分的细节：
* 对于stride=1的层，保持了输入和输出featrue-map大小的一致性，从而方便实现eltwise+的操作；
* 除了第一层和最后一层之外，便在没有pooling-layer了，下采样是通过stride=2来实现的。
* redisual-34的计算复杂度仅仅为3.6 billion m-Adds，是VGG19的18%(19.6 billion m-Adds)
* 下图的实线 shortcut 表示identity-shortcut。虚线表示当eltwise双方维度不等时，所表示的shortcut。其中维度不等有两种途径解决：
    * 通过补0来保证维度一致，从而保证identity-shortcut；
    * 利用1x1卷积核来升维。但是这需要额外的参数；

<figure>
	<a href=""><img src="/assets/images/residual/34layer.png" alt="" width="300" heigh="600"></a>
</figure>

## Bottleneck结构

论文中有两种residual-block的设计，如下图所示：
<figure>
	<a href=""><img src="/assets/images/residual/block2.png" alt="" width="400" heigh="400"></a>
</figure>

当训练较浅的网络时，我们会选用前面的这种；而如果网络较深时，会使用右图的结构（bottleneck）；之所以叫这种结构为bottleneck，是因为
这种结构**两头大，中间小**。网络输入时很大，但是先通过一层1x1卷积层来降维，减少3x3卷积的计算量。之后再利用1x1的卷积进行升维，实现
卷积前后维度不变的目的；两者的计算量是在同一个量级上，但是**bottleneck结构有三层，而原本的结构只有两层**，起到了网络更深（充数）的作用；


## Identiy mapping的探索
通篇residual-network，其实并没有严谨的告诉我们为什么residual-block的堆砌能够使得网络变得很深。在这篇论文[Identity Mappings in Deep Residual Networks](Identity Mappings in Deep Residual Networks)中，作者做了详尽的解释。并且提出了进一步的改进。提出了以下的Residual-Unit结构。
<figure>
	<a href=""><img src="/assets/images/residual/block3.png" alt="" width="500" heigh="500"></a>
</figure>

至于为什么要这么改进，下面将详细的进行解释;

在这篇论文中，作者将通用的Residual Unit公式化为一下计算：
$$
y_l = h(x_l) + F(x_l, W_l) \\
x_{l+1} = f(y_l)
$$

其中：以上几个符号需要说明一下：
* $$x_l$$表示第$$l$$个 Residual Unit 的输入。$$y_i$$表示第$$l$$个 Residual Unit 的输出；
* $$h$$表示某个变化，在这里一般是指恒等变化。但是作者在论文中探索了函数$$h(x)$$的几个不同形式；
* $$f$$代表某种操作，一般认知了指代ReLU+BN操作；

### Identiy-mapping

在上面的公式中，假如我们认为函数$$f, h$$均为identity-mapping。那么上面的公式就可以写成：

$$
x_{l+1} = x_l + F(x_l, W_l) \\
x_L = x_l + \sum_{i=l}^{L-1}F(x_i, W_i),
$$

从上面的公式，我们可以看到，经过一系列的递推后，网络有一些优秀的性质：
* 每一层的$$x_l$$，可以直接通过加上残差项，直接的forward到任何一层$$x_L$$;
* $$x_L$$是一个累加的结果，对比于传统的卷积层级联，是一个连乘的结果；

同样优秀的性质可以在反向传播的求导过程中看到：

$$
\frac{\partial{\epsilon}}{\partial{x_l}} = \frac{\partial{\epsilon}}{\partial{x_L}}\frac{\partial{x_L}}{\partial{x_l}} = \frac{\partial}{\partial{x_L}}(1 + \frac{\partial}{\partial{x_l}}\sum_{i=l}^{L-1}F(x_i, W_i))
$$

* 任何一层的导数$$\frac{\partial}{\partial{x_L}}$$都可以通往到浅层导数组成中；
* 另外一项同样也是一个累加项；

ResidualNetwork为什么能够这么深，正是这种优秀的传递形式，**使得网络中的每层信息能够随意、原封不动的流动。保证的训练过程的有效性。**

是否恒等变化才有此优秀的性质呢？答案是Yes！当如果函数$$h(x) = \lambda x$$时（不再是恒等变化）。我们推出类似上面结构的结果：

$$
x_L = (\prod_{i=l}^{L=1}\lambda_{i})x_l + \sum_{i}^{L-1}F(x_i, W_i)
$$

同理于求导过程（具体可以参考论文）。每一层的信息不再是原封不动的forward到任意的层，而是前面有了一个参数$$\lambda$$的连乘。当层数过多时，这时候就会容易发生
梯度弥散($$\lambda < 1$$)或者梯度爆炸等现象($$\lambda > 1$$)

### Usage of Activation Functions
论文同时还探索了激活函数ReLu的摆放位置对结构的影响(也就是前面的如何将函数$$f$$变成恒等映射)。我们可以来看下图：
<figure>
	<a href=""><img src="/assets/images/residual/pre_activation.png" alt="" width="800" heigh="800"></a>
</figure>

可以看到，在上图作者对cifar10进行的多组实验中，使用full pre-activation这种Residul Unit效果最佳，个人认为这张表格还是挺重要的，我们简单分析一下！

* **BN after Addition：** 作者尝试将BN层移到主干路上，发现效果很差。因为BN层对结果的scale和offset影响了恒等映射，这阻碍了信息的传递。
* **ReLu before Addtion：** 将函数$$f$$变成恒等映射，最容易想到的就是将Addtion后的ReLU移动到BN后面。但是不要忘记的一点事。$$F$$函数学习的
是一个残差函数，而残差函数的输出应该是在范围$$(-\infty, +\infty)$$之间，而不应该被限定大于0。这影响了残差函数原有的功能，因为效果也不太work。

直接提上来似乎不行，但是问题反过来想， 在addition之后做ReLU，不是相当于在下一次conv之前做ReLU吗？

* **ReLU-only pre-activation：** 根据刚才的想法，我们把ReLU放到前面去，然而我们得到的结果和(a)差不多，原因是什么呢？因为这个ReLU层不与BN层连接使用，因此无法共享BN所带来的好处。
* **full pre-activation：** 啊，那要不我们也把BN弄前面去，惊喜出现了，我们得到了相当可观的结果，是的，这便是我们最后要使用的Unit结构！！！

## 杂谈

ResidualNet的发明开创了CNN结构上一扇新的大门。对后续的深度网络设计产生了极大的影响。在最后再简单说几个有趣的事情：

* [Residual Networks are Exponential Ensembles of Relatively Shallow Networks](https://link.jianshu.com/?t=https://arxiv.org/abs/1605.06431)
这篇文章指出ResNet其实并不是真正意义上的深层网络，它只是若干浅层网络的组合形式。具体的分析，可以看看论文本身和这篇博客：[ResNet到底深不深？](https://www.jianshu.com/p/b724411571ab)

* 为什么Residual Unit中一般是跨2，3层的卷积层呢？ 一层行不行？其实论文中也有提到，当只有一层时$$y = Wx + x$$这种形式更偏向于一个线性的结构，并没有很好的优势。同时，知乎上有一篇[实验](https://zhuanlan.zhihu.com/p/26595791)，也论证了为什么两层网络的堆叠才work。

* Residual Unit中，Caffe主要利用 `eltwise-layer`来实现eltwise+的操作。具体也可以看看代码[eltwise_layer.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/eltwise_layer.cpp)
