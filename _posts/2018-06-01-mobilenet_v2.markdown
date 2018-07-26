---
layout: post
title: "卷积神经网络-快而小的网络：MobileNetV2"
modified:
categories: DeepLearning
description: "卷积神经网络-快而小的网络总结：MobileNetV2"
tags: [cnn dl model_compression]
comments: true
mathjax: true
share:
date: 2018-06-01T13:54:50+08:00
---

本文介绍的论文[MobileNet V2](https://arxiv.org/abs/1704.0486://arxiv.org/pdf/1801.04381.pdf)是
基于 MobileNetV1 的改进。 在 MobileNet 中，提出应用 Depthwise-separaable 卷积 和 两个超参来控制网络的计算量。
在保持移动端可接受的模型复杂度的基础上达到了相当的精度。

在上一篇[博客](http://dreamingo.github.io/2018/05/small_network1/#mobilenet)
中我们也提到，MobileNetV1这种比较复古的直筒结构，是Google三年前的工作了。MobileNetV2将 Residual shortcut 的那一套引入到
小网络的设计中，并且提出 **Inverted residual with linear bottleneck** 的结构进一步来改进。同时，论文还提出了一些inference过程中内存
优化的技巧，比起一些只从理论分析计算量的网络而言，是十足良心可靠的工业之作。

## Reference
+ [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
+ [知乎：如何评价mobilenet v2 ?](https://www.zhihu.com/question/265709710)
+ [[论文笔记](MobileNet V2)Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://blog.ddlee.cn/posts/c9816b0a/)


## 背景&讨论

### 深度分离卷积

MobileNetV2继续沿用了来自 MobileNetV1 中提出（最初提出应该是Xception）的`Depthwise Separable Convolutions`结构。关于DCONV的原理和复杂度分析等就不在这里介绍了，具体可以
参考上一篇[博文](http://dreamingo.github.io/2018/05/small_network1/#mobilenet)。众多的实验和文献指出，DCONV结构能够在大大减少计算量（约减少$$k^2$$倍，$$k$$为DCONV卷积核的大小）的同时，
与传统的卷积结构相比丢失的精度可以忽略不计。在近年来的工作中， DCONV大行其道，可以看做是小网络设计的标配了，对着往后网络结构的实际有着深刻的影响。

### Linear Bottlenecks

文中花了很大的力气去阐释为什么需要引入 Linear Bottleneck 的设计。在这一部分中和论文的附加资料中，都主要阐述了 RELU 函数对低维特征的破坏。

在论文中，指出经过激活层的张量都被称为『兴趣流形』（原文为*manifold of interest*，这个翻译有些尴尬）。论文中指出，manifold of interest分布在高维空间中的一个低维
流形中（关于机器学习中的流形，可以参考[知乎回答-有谁给解释一下流形以及流形正则化？](https://www.zhihu.com/question/40794637/answer/88455767)）。而正是因为这些激活的
输出值是分布在低维的流形中，因此在类似 SqueezeNet 1x1 SqueezeLayer & MobileNet 1x1 卷积 和 width multiplier 应用中，可以将输入数据进行降维压缩。但是文章指出，这些压缩
方法（1x1卷积）后接的RELU，在某些情况下会对特征造成大规模不可逆的损坏。

究竟是在什么情况下，RELU会对特征造成大规模不可逆的损坏呢？首先ReLU将 <0 的值置为0，输入就会变得稀疏。若所有输入的某一个维度都被置0了，则会令输入空间本身降维；
论文实验指出，当数据的维度较高时，能够有效的保存和恢复特征。因为RELU会令部分的channel坍塌(Collapses)掉，从而造成信息的不可逆丢失。但是如果channel数目足够的多，那么部分channel坍塌后，其他的channels锁还是有足够的信息来表示这个低维度的『流形结构』。

<figure>
	<a href=""><img src="/assets/images/mobilenet_v2/relu.png" alt="" width="700" heigh="700"></a>
</figure>


上图是是论文中针对 『RELU会对低维特征造成大规模不可逆损坏』的一个实验印证。在上面的实验中，主要分为以下几点：
1. 原始数据 *input* 是一个2维的数据$$x \in R^{k \times 2}$$($$k$$是数据点的个数)；可以看到上图的图1；
2. 矩阵$$T \in R^{2 \times n}$$是映射矩阵，能够将原始数据 *input* 映射到高维数据上；
3. 因为我们假设 高维数据$$xT \in R^{k \times n}$$是嵌套在低维（2维）流形上的子空间中，所以我们对$$x$$进行升维，得到其高维的表示$$xT$$;
4. 我们将高维表示进过RELU操作后，利用$$T^{-1}$$映射回2维空间中：$$x' = RELU(xT)T^{-1}$$
5. 将重新映射后的信息$$x'$$画出来；

从实验图我们可以看到，当$$n$$比较小时，manifold的信息丢大规模的丢失了。而当$$n$$较大时，manifold的信息得到有效的保存。因此，这个
实验得出了一个重要的结论（这也是论文中一个重要的改善点：linear-bottleneck）：

1. **对维度较低的张量表示没有必要应用RELU，因为利用RELU进行变换会导致很大的信息损耗**
2. **在需要使用RELU的卷积层中，可以先利用1x1矩阵对矩阵进行升维，再应用RELU激活函数**

## 改进 & Why

其实上面铺垫了那么多，那么久，是时候直接指出 MobileNetV2 比起V1版本的改进，以及**为什么**这么做：

<figure>
	<a href=""><img src="/assets/images/mobilenet_v2/block.png" alt="" width="700" heigh="700"></a>
</figure>

1. **引入了 Inverted-Residual 结构**：首先类似 residual unit 中引入了 `shortcut eltwise+` 的操作，起到了复用特征的作用。其次，传统的residual-block是
『两头大，中间小』的结构（因为没有采取DCONV的方式，因此需要先1x1降维，减少3x3卷积的计算量）。而 Inverted-Residual 则是一个相反的方式『两头小，中间大』，
第一层的1x1卷积起到了升维的作用，而后面的1x1卷积则起到降维压缩的作用；
2. **Linear Bottleneck：**去掉了最后一个用于降维的1x1卷积后的ReLU6激活函数；

而为什么要引入这两个改进了？**Why**？原因其实在上面都已经呼之欲出了，在这里再总结一下：

Inverted-Residual的引入主要有两个作用：
* 比起MobileNetV1的这种直筒结构，引入`shortcut eltwise+`操作，在不影响计算量的情况下能够复用特征、方便深层模型训练等作用。简直是free-lunch；
* 之前指出ReLu会有对低维特征破坏的情况。因此Inverted-Residual中引入1x1来升维（Expand-layer），再接DCONV+ReLU6来抽取特征；通过增加维度，来减少ReLU对特征的破坏；

Linear Bottleneck的引入，主要原因也是1x1降维后，去掉ReLU操作来减少特征退化的情况。

## 实现细节

### 内存优化

文章中除了大量的理论分析，还引入了一些工程上关于如何优化inference过程中内存使用的问题，十足的良心。在这里先引用一个知乎上对 [Alan Huang的回答：如何评价MobileNetV2](https://www.zhihu.com/question/265709710/answer/298245276)：

> **效率优先的网络结构设计**：以前大多数的模型结构加速的工作都停留在压缩网络参数量上。 其实这是有误导性的： 参数量少不代表计算量小； 计算量小不代表理论上速度快（带宽限制）。 ResNet的结构其实对带宽不大友好： 旁路的计算量很小，eltwise+ 的特征很大，所以带宽上就比较吃紧。由此看Inverted residual 确实是个非常精妙的设计！   其实业界在做优化加速， 确实是把好几个层一起做了， 利用时空局部性，减少访问DDR来加速。 所以 Inverted residual  带宽上确实是比较友好。 所以总结一下： 这个确实是非常良心， 而且实用靠谱的东西，业界良心。

刚看到这个回答的时候，对其中的话术也是似是而非的明白一些，随着和作者的交谈和进一步细读论文后，我尝试做出一些解释。

#### **中间内存优化**
在上面的回答中，我们可以看到有这么一句话**『其实业界在做优化加速，确实也是好几个层一起做了』**。这句话在论文中也类似的说法：

> *This convolutional module is particularly suitable for mobile designs, because it allows to significantly reduce the memoery footprint needed during inference by never fully materializing large intermediate tensors*

论文中指出，一个bottleneck-block的计算$$F(x) = [A \circ N \circ B]$$,其中：

* A是第一个1x1卷积expanded-layer， $$A: R^{s \times s \times k} \rightarrow R^{s \times s \times n}$$
* N是中间的非线性dconv层，$$N: R^{s \times s \times n} \rightarrow R^{s' \times s' \times n}$$
* N是可以表示为：$$N = ReLU6 \circ dwise \circ ReLU6$$
* B是第一个1x1卷积， $$B: R^{s' \times s' \times n} \rightarrow R^{s' \times s' \times k}$$

其中我们考虑到，因为**`DCONV`是逐个channel做的，结果是每层的结果cancat起来的**；假设我们分 $$t$$ 次来做（极限情况下$$t = n$$，就是逐层）。那么理论上A层每产生 $$n/t$$ 个channels，N层就可以做对应的操作。而B层也可以将N层的输出先
部分计算到输出中。在这种情况下，所需的内存数量是$$|s^2k| + |s'^2k| + O(max(s^2, s'^2))$$

上面公式的第一项是输入特征所占内存的大小（因为之后还有做`eltwise+`操作），第二项是输出的特征图大小（一方面是存储B层的计算结果，同时用于`eltwise+`操作），而第三项则是DONV所需的内存占用；

因此，这就印证了论文开始的那句话。对于中间的tensor，我们没有必要全部生成并且展开（而且很大），只需要用一些**『复用』**的中间层来存储结果就可以了。这样一来就大大减少了运行过程中的内存使用；

但是，文中也有指出，我们将计算过程拆分成$$t$$个独立的过程，并不会对理论的计算量有任何改变。但是**实际的运行时间却增大了**，因为将一个大的矩阵乘法运算拆分成若干个小的运算会导致 cache-miss 的增加。因此，文中建议，$$t$$的取值比较小为好（2-5之间），这样一来即大大的减少的内存的需要，也有效的利用了高度优化的矩阵乘法计算。

<figure>
	<a href=""><img src="/assets/images/mobilenet_v2/memory.png" alt="" width="700" heigh="700"></a>
</figure>

#### **eltwise带宽优化**

在上面的回答中，我们还主要到这么一句话：**『ResNet的结构其实对带宽不大友好： 旁路的计算量很小，eltwise+ 的特征很大，所以带宽上就比较吃紧。』**。这个是什么意思呢？

从上面的内存数量分析我们可以看到：$$|s^2k| + |s'^2k| + O(max(s^2, s'^2))$$。由于中间的结果我们采取复用的形式，因此中间结果的所占的内存极少。重点在于eltwise+操作时特征的大小($$s^2k$$)。
由于residual-unit的bottleneck结构（两头大，中间小），大的两头连接的eltwise操作会导致带宽很大。而 inverted-residual 这种精妙的设计，两头小中间大，eltwise操作连接的两头中间都很小，因此带宽占用小。

### 网络结构

下图中，$$t$$ 代表单元的扩张系数，$$c$$ 代表channel数，$$n$$ 为单元重复个数，$$s$$ 为stride数。可见，网络整体上遵循了重复相同单元和加深则变宽等设计范式。也不免有人工设计的成分（如$$28^2 \times 64$$单元的stride，单元重复数等）。
<figure>
	<a href=""><img src="/assets/images/mobilenet_v2/network.png" alt="" width="700" heigh="700"></a>
</figure>


### ReLU6
论文中使用的激活函数都是ReLU6，其数学表示是：
$$
ReLU6 = min(ReLu(x), 6)
$$

这么做的目的是为了在移动端设备使用低精度float16时，也能有很好的数值分别率。如果不加限制，激活数值很大时，则低精度的float16无法很好的精确描述，带来精度损失。


## 结语

MobileNetV2在精度上比起V1有了蛮大的提升，在同精度下计算量(mAdds)也比V1要少一些。在Google自己[开源的代码和数据](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)中和[腾讯的ncnn框架](https://github.com/Tencent/ncnn/tree/master/benchmark)的benchmark结果来看，都比V1快了一些；

<figure>
	<a href=""><img src="/assets/images/mobilenet_v2/time.png" alt="" width="700" heigh="700"></a>
</figure>
