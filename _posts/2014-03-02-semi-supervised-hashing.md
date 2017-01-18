---
layout: post
title: "论文笔记:Semi-Supervised Hashing"
modified:
categories: 机器学习与数据挖掘
description:
tags: [Machine Learning  Hashing Semi-Supervised]
comments: true
mathjax: true
share:
---


此文主要对论文[_Semi-Supervised Hashing for large Scale Search_](http://sanjivk.com/SSH_PAMI_final.pdf)的理解以及整理。

### Introduction

在信息爆炸的时代，基于图片的检索（Content based image retrieval CBIR）得到了大量的需求和关注。通常在CBIR中，给出一个visual query q，返回与其**大约**（权衡效率和精度的情况下，没必要一定要找最相近的）最相近的邻居集合（ANN - Aproximate nearest neighbors）。

在传统的方法中，基于树形数据结构的方法（如KD树，球树等）有着优秀的检索速度（$$O(log(n))$$）。然而，这些基于树形的方法往往在高维的数据上性能急剧下降（最坏情况是下降到linear search的地步）。除此之外，这些tree-based的方法往往还会遇到内存占用过大的问题，因为很多情况下，用于存储的数据结构本身的大小甚至大于元数据本身。基于哈希的ANN技术因此得到更多的关注。

在哈希技术中，我们通常用压缩的哈希码来表示一个样例。通常，给出一个n D-dim的向量$$\textbf{X} \in R^{D * n}$$，通过K个哈希函数，来学习出一个K-bit的二元码
（$$\textbf{Y} \in B^{K * n}$$）来表示所有样例。线性投影的哈希技术（Linear projection-based hasing）是最常用的方法之一。通常第k个哈希函数表示为：

$$
h_{k}(\textbf{x}_{i}) = sgn(f(\textbf{w}_{k}^{T}\textbf{x} + b_{k})),
$$

在上式中，$$\textbf{x}$$是一个数据点，$$\textbf{w}_{k}$$则称为投影向量（projection vector）。由于$$h(x) \in \{-1, 1\}$$，而哈希码$$y \in \{0, 1\}$$。因此一般可以表示为$$y_{k}(\textbf{x}) = (1 + h_k(\textbf{x})) / 2$$。对于$$\textbf{w}$$和$$f(.)$$的不同选择，会导致不同的哈希方法。

论文中简单了介绍了现今的几种有效的哈希方法，对于各式各样的哈希方法，一般分为三大类：

* Unsupervised methods（数据集中没有标明彼此之间的相似性）

    * LSH [Locality Sensitive hasing]
    * Spectral hashing
    * Graph based hasing ...
    

* Supervised method

    * Boosted Similarity Sensitive Coding [BSSC]
    * Restricted Boltzmann Machines [RBM]...


* Semi-Supervised method（只有一部分数据集标明了label或者彼此的相似性）

本文主要介绍Semi-Supervised learning method

<!--more-->

### Semi-Supervised Paradigm For Hashing

#### [Definition] 

在这里，首先给出一些相关的定义：首先，给定n个数据样本，$$\chi = \{ \textbf{x}_{i} \} $$, $$i$$ = 1...n; $$\textbf{x}_{i} \in R^{D}$$。并且，其中的一部分数据样本
标注了相关的类别信息$$\mathcal{M}和\mathcal{C}$$。其中，如果$$(\textbf{x}_{i}, \textbf{x}_{j} ) \in \mathcal{M}$$，则表示他们是相邻的一对。同理，如果$$(\textbf{x}_{i}, \textbf{x}_{j} ) \in \mathcal{C}$$，则表示他们不太相似，或者有不同的类别标签。假设其中有$$l$$个数据点属于$$\mathcal{M}或者\mathcal{C}$$，则可以定义$$S$$矩阵（$l$ × $l$）如下：

<img src="{{site.url}}/images/semi_supervised_hashing/S_matrix.png" width="400px"/>

#### [Empirical Fitness]

在之前，我们已经定义过相关的哈希函数的形式以及哈希码的格式了：

$$
h_{k}(\textbf{x}_{i}) = sgn(f(\textbf{w}_{k}^{T}\textbf{x} + b_{k})),
$$

一般在训练过程中，会将数据$$\textbf{X}$$做normalized处理，使其平均值为0。而在上式中，$$b_{k}$$表示投影数据的平均值。因此，在这里，$$b_{k} = 0$$

$$
y_{ki} = \frac{1}{2}(1 + h_{k}(\textbf{x}_{i})) = \frac{1}{2}(1 + sng(\textbf{w}_{k}^{T}\textbf{x}_{i}))
$$

将$$\textbf{H} = [h_{1} ..., h_{k}]$$表示为K个对应的哈希函数，而$$\textbf{W} = [w_{1} ... w_{K}]  \in R^{D × K}$$。在这里，我们希望通过学习出一个$$\textbf{W}$$，对于哪些$$(\textbf{x}_{i}, \textbf{x}_{j} ) \in \mathcal{M}$$，学习出相同的bits，而对于那些$$(\textbf{x}_{i}, \textbf{x}_{j} ) \in \mathcal{C}$$的样例，则最好学出不同的bits。因此，根据经验风险最小化的原则，我们定义其目标函数为：

<img src="{{site.url}}/images/semi_supervised_hashing/j_function.png" width="500px"/>

而我们想要的，则是得到一些列的哈希函数$$\textbf{H}$$， 以此最大化目标函数。不是一般性的，我们可以将其表示为矩阵的形式（其中，tr表示矩阵的trace，一般为方阵的对角线元素之和）：

$$
\begin {eqnarray}
J(\textbf{H}) & = & \frac{1}{2}tr\{ \textbf{H}(\textbf{X}_{l}) \; \textbf{S} \; \textbf{H}(\textbf{X}_{l})^{T}\}  \\\
& = & \frac{1}{2} \textbf{tr}\{sgn(\textbf{W}^{T}\textbf{X}_{l})\; \textbf{S} \; sgn(\textbf{W}^{T}\textbf{X}_{l})^{T} \} \\\
\end {eqnarray}
$$

而上式中，由于sgn函数的存在，导致目标函数不可导。其实，我们完全可以在目标函数中去掉sgn函数的作用。变成了一个带符号大小的度量：当两个样例相近时，
不仅要求其符号相等，更要求其投影后也相近，而两个样例不相似时，要求其投影后符号不同，而且其大小也相差越大越好。因此，我们可以将目标函数改为：

$$
J(\textbf{W}) =  \frac{1}{2} \textbf{tr}\{\textbf{W}^{T}\textbf{X}_{l}\; \textbf{S} \; \textbf{X}_{l}^{T}\textbf{W} \} \\\
$$

#### Information Theoretic Regularization

仅仅对这些有标记的小量数据做经验最小化的训练会导致严重的overfitting。为了得到泛化能力更好的模型，我们应该对目标函数加入正则化项。在正则化中，我们可以
运用的不仅仅是标记的数据，也可以运用无标记的数据（Semi-Supervised learning paradigm）。

从信息论的角度来看，我们希望得到的哈希码能够最大化表示更多的信息，更多的信息以为这模型的泛化能力也就越好。因此，根据[最大熵理论](http://en.wikipedia.org/wiki/Principle_of_maximum_entropy)，当这个
binary bit能够将$$\textbf{X}$$平衡的划分时（也就是哈希函数$$h_{k}$$，对于样本$$\textbf{x}$$，有0.5的几率令$$h_{k}(\textbf{x}) = 1$$，有0.5的几率令其为0），能够得到最大的熵值（更多的不确定性，更多的信息量）。在这里可以直接表示为
$$\sum_{i=1}^{n} h_{k}(\textbf{x}_{i}) = 0$$。而在这个哈希问题中，对于这个binary bit的最大熵划分，等于其的最大方差的划分（其证明过程可以参考论文）。因此，我们的正则化项可以表示为：

$$
R = \sum_{k} var[h_{k}(\textbf{x})] = \sum_{k}var[sgn(\textbf{w}^{T}\textbf{x})]
$$

由于sgn函数的存在导致该项再一次不可导，因此，消除sgn函数的作用域并且一些列的证明后，得到最终的正则化项：

<img src="{{site.url}}/images/semi_supervised_hashing/r_function.png" width="500px"/>

并且得到最终的目标函数：

<img src="{{site.url}}/images/semi_supervised_hashing/objective_function.png" width="500px"/>

### Projection Learning

接下来，将介绍三种方法来学习投影矩阵$$\textbf{W}$$

#### Orthogonal Projection Learning

我们希望学习出来的哈希码，每个bit能够包含足够多的信息，并且希望没有冗余的bit（尽可能减少哈希码的位数），一个有效的方法是是的投影的方向相互正交
（类似与PCA分解，使得投影方向相互正交从而投影出来的数据方差最大，信息量足够大）：

$$
\textbf{W} = argMax J(\textbf{W}) \\\
subject\;to\; \textbf{W}^{T}\textbf{W} = \textbf{I}
$$

这样，学习正交化的$$\textbf{W}$$就变成了一个典型的特征值分解问题。因此，我们可以对矩阵$M$做特征值分解：

$$
\begin {eqnarray}
J(W) & = & \frac{1}{2}tr\{W^{T}(W\Lambda W^{T})W\} \\\
& = & \frac{1}{2}tr\{I\Lambda I\} \\\
& = & \sum_{k}^{K}\lambda_{k} \\\ 
\end {eqnarray}
$$

其中，$$\lambda_{1} > \lambda_{2} > ... > \lambda_{K}$$，并且$$W = [e_{1} ... e_{K}]$$，其中$$e_{k}$$表示对应的特征向量。

#### Non-Orthogonal Projection Learning

在上面的一种方法中，我们要求投影方向正交化从而是每个bit的信息量足够的多。然后，在实际应用上，数据集的大部分方差只是集中在很少的几个方向（例如在PCA分解后，$$\lambda_{1}远大于\lambda_{2}$$，所有特征值的总和主要由前几个特征值贡献，导致方差只是几种在几个方向上）。因此，我们可以将[投影方向正交化]这个限制去掉，令其选择新方向的时候，不一定要正交于之前的方向，而是选择较大着。而投影正交化这个约束则可以转化成一个乘法因子。使得新的约束函数如下：

<img src="{{site.url}}/images/semi_supervised_hashing/j_function2.png" width="500px"/>

新的目标函数有对非正交特性的一定容忍度（取决于正的惩罚因子$$\rho$$）。然而，以上的目标函数却是非凸函数，不像之前的方法有比较简单快捷的方法快速找到
全局最优解。为了最大化目标函数，我们首先对目标函数进行求导并且令其为0（如果矩阵运算薄弱可以参考两个文章[Properties of the Trace and Matrix Derivatives](http://www.cs.berkeley.edu/~jduchi/projects/matrix_prop.pdf), [Matrix Calculus](http://www.atmos.washington.edu/~dennis/MatrixCalculus.pdf)）

$$
\begin {eqnarray}
\frac{\partial{J(W)}}{\partial{W}} & = & (W^{T}.M)^T - 2\rho(W^{T}.W - I)^T.(W^T)^T = 0 \\\
& = & MW - 2\rho(WW^{T} - I).W = 0 \\\
& = & (WW^{T} - I - \frac{1}{\rho}M) = 0 \\\
& = & WW^{T}W = (I + \frac{1}{\rho}M)W = 0 \\\
\end {eqnarray}
$$

论文中证明了当矩阵$$Q = I + \frac{1}{\rho}M$$是正定时，$$Q$$可以通过 Cholesky decomposition $$Q = LL^{T}$$。因此，我们可以得到$$W = LU$$，其中U矩阵为M矩阵的特征向量组成的矩阵。为了减少运算量和提高精度，我们可以只是取W的前k列。其中$$U_{k}表示M的前k个特征向量$$。得：

$$
W = LU^{k}
$$

#### Sequential Projection Learning

对于第三种求投影向量的方法，这里就无力叙述了。方法比较直观明了（类似于boost方法），大家可以直接参考论文。

