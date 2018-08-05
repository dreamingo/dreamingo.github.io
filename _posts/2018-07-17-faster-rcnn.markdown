---
layout: post
title: "Object-Detection系列 - Fast&Faster R-CNN"
modified:
categories: 机器学习 
description: "Object-Detection Deep-Learning RCNN"
tags: [Object-Detection Deep-Learning RCNN]
comments: true
mathjax: true
share:
date: 2018-07-17T01:06:13+08:00
---

## 0x01. 前言
<figure>
	<a href=""><img src="/assets/images/yolo/big_pic.png" alt="" width="600" heigh="200"></a>
    <figcaption><a href="" title="">目标检测算法一览概况</a></figcaption>
</figure>

在前面，我们介绍了基于CNN系列的开山之作RCNN，也列举了 RCNN 算法的众多缺点。在这篇文章中，我们重点介绍 Faster-RCNN 的工作，**由于 Faster-RCNN 工作中基本包括了所有 Fast-RCNN 工作中的亮点，**，所以这里就两者一起介绍好了。

在这里，我们再次列举下 RCNN 工作的**痛点**，同时也简单的介绍下 Faster/Fast RCNN 系列是如何改进的：

> 1. 训练、预测过程多分为多个阶段，步骤繁琐并且引入一定的误差；
> 2. 训练耗时，同时模型中间保存的特征结果占据极大的磁盘空间（5000张图片会产生200G的特征文件（5000 * 2000 * 4096 * 4））；
> 3. 预测速度极慢，完全达不到工业的标准。VGG16模型利用GPU来预测一张照片需要47s。

而在 Faster RCNN 工作中，对以上痛点进行了**分析和改进**：

1. Fast-RCNN 论文指出，RCNN预测过程之所以慢，是因为其需要对2000多个 region-proposal 区域进行**重复的特征提取操作**。其通过 **`RoI-Pooling-Layer`** 解决了这个只对原始图片进行一次特征提取的过程，将每个region-proposal 映射到 feature-map 上对应的特征块，从而实现同一张图中不同 proposal **共享计算**的问题，大大的加快了算法过程。
2. Faster-RCNN 提出了 RPN（Region-Proposal-Network），与 Fast-RCNN 网络共享同一张 feature-map，实现了**端到端**的 proposal提取、分类和 bbox 校准回归的过程。将所有的阶段一体化。
3. 由于 region-proposal、特征提取、分类、bbox回归等操作连贯在一次，因此模型基本不需要有中间结果的参数，**节省了训练过程中所需的磁盘空间**；
4. 如上图所示，无论是**预测速度和精度上都有了极大的提升**。

## 0x02. 整体模型概述

论文中提到，Faster-RCNN 是一个端到端、一体化的目标检测系统。整个系统可以分为两大部分：

* **RPN(Region Proposal Network)：** 该网络接受一个原始的图片，并输出该图片的 region-proposals以及其属于 foreground(前景) 还是 background(背景) 的概率。
* **Fast-RCNN：** 该部分基本完全沿用 Fast-RCNN 的工作，该网络输入接受一个原始的图片及其 region-proposals，同时对 bounding-box 执行分类和回归校准的工作。

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/faster-rcnn.png" alt="" width="300" heigh="300"></a>
</figure>

从上图中，我们可以看到，如果对整体模型作进一步细分，可以分成4个部分：

1. **Conv-Layer:**  卷积层包括一系列卷积(Conv + Relu)和池化(Pooling)操作，用于提取图像的特征(feature maps)，一般直接使用现有的经典网络模型ZF或者VGG16，而且卷积层的权值参数为 RPN 和 Fast RCNN 所**共享**，这也是能够加快训练过程、提升模型实时性的关键所在。
2. **Region Proposal Network:** RPN 网络用于**生成区域候选框 Proposal**，基于网络模型引入的多尺度Anchor，通过`Softmax`对 anchors 属于目标(foreground)还是背景(background)进行分类，并使用Bounding Box Regression 对 anchors进行回归校准预测，获取Proposal的精确位置，并用于后续的目标识别与检测。
3. **RoI Pooling:** 综合卷积层特征feature maps 和候选框 proposal 的信息，将propopal在输入图像中的**坐标映射**到最后一层feature map(`conv5-3`)中，对feature map中的对应区域进行max-pooling操作，得到**固定大小**(7×7)输出的池化结果，并与后面的全连接层相连。
4. **Classification and Regression:** 全连接层后接两个子连接层——分类层(cls)和回归层(reg)，分类层用于判断Proposal的类别，回归层则通过bounding box regression预测Proposal的准确位置。

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/art.png" alt="" width="700" heigh="300"></a>
    <figcaption><a href="" title="">Faster-RCNN架构图</a></figcaption>
</figure>

## 0x03. Conv-Layer：

从下图中我们可以看到，Faster-RCNN模型中是依赖于下图的卷积层结构来抽取特征。该卷积网络共包含13个卷积层，4个`max-pooling-layer`。其中值得注意的是，**这13个卷积层均为 $$3 \times 3$$，padding=1，stride=1 的卷积核，，这意味着特征图经过该卷积层作用后，大小不变。整个模型的下采样(Down-Sampling)工作是通过 pooling 层来完成的。**

这一点之所以很重要，是因为在后续过程中，我们会从原始输入图片的坐标推导到`conv5_3` 层特征图上的坐标，由于整个模型只有 `pooling` 层引入了下采样，因此可以通过缩放因子$$2^4 = 16$$，来从原始图片到特征图的映射。这一部分在后面 `RoI-Pooling-Layer` 会再仔细说说。

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/conv.png" alt="" width="500" heigh="200"></a>
    <figcaption><a href="" title="">Conv-Layer部分</a></figcaption>
</figure>

在上图中，我专门用红框来突出了，在输入模型前，算法会将图片 resize 成 $$M \times N$$的大小。这一点中在代码可以看出（图片的长边要求不超过1000，短边要求不超过600）：

{% highlight python %} 
# 代码路径:py-faster-rcnn/lib/fast_rcnn/test.py
for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # MAX_SIZE = 1000
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)
{% endhighlight %} 

## 0x04. Region Proposal Network(RPN)

RPN 网络提出的初衷，是因为传统的候选框生成算法（如Seletive-Search）一方面作为额外算法独立于目标检测系统，破坏了系统的一体化进程；另一方面其运行速度慢（只能运行在CPU上，2s/image的处理速度），往往会成为整个目标检测系统中速度的瓶颈。
在 Faster-RCNN 中则提出 RPN 网络，将检测框 Proposals 的提取嵌入到深度卷积网络中，一方面可以利用GPU加速计算，另一方面通过共享卷积参数的方式有效的减少计算量，从而提升了 Region-Proposal 的生成速度。

### 全卷积网络
RPN网络是一个**全卷积网络（Full-Convolutional Network）。** 在开始介绍 RPN 网络前，我想先简单介绍下全卷积网络的应用。在以往的工作中，我们常常会看见在卷积网络的最后会添加若干个**全连接层**来进行回归或者分类的工作。但是近期来一些性能优异的网络(ResNet, GoogleNet)都会利用 GAP（Global Average Pooling）技术来取代FC来对融合得到的深度特征，在减少参数冗余的前提下还能获得不错的性能。

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/gap.png" alt="" width="500" heigh="300"></a>
    <figcaption><a href="" title="">先利用卷积操作将通道数目转化为和分类数目一致，相当于每张feature-map等于每个类别的概率图，再利用GAP得到输出的概率</a></figcaption>
</figure>

在 RPN 网络中，网络最后的回归和分类工作，都利用了 1x1 卷积核来代替全连接层进行特征融合。以该网络(VGG16)的分类任务为例，对于 $$ H \times W \times 512$$ 的卷积层，利用 $$512 \times 1 \times 1 \times 18$$ 的卷积核对通道信息进行整合，输出的 18 个通道，**相当于同时训练了 9 个二分类分类器。**


### Anchor 机制

我们都知道，RPN 网络是负责生成 region-proposals 的。那么在 PRN 网络中是如何生成可能存在物体的检测框的呢？由于基于滑动窗口的方法一来效率会非常低下，而来会生成非常多重叠度非常高无用的边框。因此论文借鉴了 SPP 和 RoI 中的思想，反过来从卷积层的 feature-map 中提取 proposal。

网络在最后一个卷积层的特征图 `conv5_3` 上，针对该特征图上的每一个点，利用一个 $$ 3 \times 3$$ , $$padding = 1, stride=1$$ 的卷积核进行滑动。**对于这个操作，可以分开两个方向来看**：

**1. 向网络底部方向：** 算法在卷积滑动过程中，每个卷积核的**中心点**映射回输入图像中。为了实现**多尺度**的检测，在原图的映射点上，生成3中尺度(scale) $$\{128^2, 256^2, 512^2\}$$ 和 3中长宽比(ratio) $$\{1:1, 1:2, 2:1\}$$ 共9个检测框。**在算法中，对于feature-map上的每个点，这9个检测框叫做该点的 Anchor-boxes。**
<figure>
	<a href=""><img src="/assets/images/faster-rcnn/box.png" alt="" width="500" heigh="300"></a>
    <figcaption><a href="" title="">对conv5_3中的每个点生成9个不同尺寸的anchor-box</a></figcaption>
</figure>

**2. 向网络顶部方向：** 在大小为 $$H \times W \times 512$$的`conv5_3`经历$$ 3 \times 3 \times 512$$ 卷积(+ReLU)后，生成大小依然为 $$ H \times W \times 512$$的特征图 `rpn\output`。在这里，相当于原 `conv5_3` 中的每个$$ 3\times 3$$的滑动窗口被映射成 $$512d$$ 的实数向量。这时候网络会发生分叉：
* 利用 $$1 \times 1 \times 18$$ 的卷积核对 `rpn\output` 中的feature-map进行卷积操作，这相当于利用 $$512d$$ 的特征向量同时训练9个二分类器（$$9 \times 2 = 18$$），最后再输入到 Softmax 层中，对9个 anchor-box 进行分类（是否包含物体，属于前景/背景）；

* 利用 $$1 \times 1 \times 36$$ 的卷积核对 `rpn\output` 中的feature-map进行卷积操作，这相当于同时训练了9个 bounding-box 回归器，对每个Anchor中的坐标进行偏移量回归校准操作。

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/UP.png" alt="" width="400" heigh="400"></a>
    <figcaption><a href="" title="">对3x3卷积后的512维特征进行分类和bbox回归</a></figcaption>
</figure>

- - -

**下面，我们再针对这两个方向(UP/DOWN)，详细的介绍一下其中的细节&Why**


### Anchor-Box的生成与原理

在这里，我们展开来讨论向下传播过程中的Anchor-Box的生成细节。对此，我们直接来抛出几个问题，逐个解答就好了。

> 1. 这种方法为什么能够检测出 region-proposal？好处在哪？如何直观的理解该算法？
> 2. 特征图上的点是如何传播到原图上的，如何计算特征图上的点到原图的感受野(Receptive Field)？
> 3. 这9个Anchor-boxes的大小都超出了这 3x3 卷积的感受野，检测还能有效吗？

<span style="color:#1A2F7B;font-size:120%"><strong>1.这种方法为什么能够检测出 region-proposal？好处在哪？如何直观的理解该算法？</strong></span>

初看该算法，我心中当时惊叹了一下该算法**既暴力又取巧**。这其中的原因：
1. 因为原图像素多尺寸大($$1000 \times 600$$)，直接在上面进行滑窗/图片金字塔(slide-windows)方式的 region-proposal 方法极其的低效。
2. 当网络经过一系列下采样后，feature-map 的尺寸变小了。就可以通过**滑窗**这种暴力的方法来解决了，一来Anchor共享卷积图计算高效，二来可以通过NMS（非最大值抑制算法）来有效去掉这种暴力方法所产生过多的boxes；
3. 每个点引入9中不同尺寸的box，来尽可能的探测不同尺寸或者解决同一位置上多个物体的问题。
4. 这样所产生的 Anchor-box 尽然不太准确。But Anyway，后面还有两次对 bounding-box 的回归校准呢？这也是其中暴力中取巧的一点。

<span style="color:#1A2F7B;font-size:120%"><strong>2. 特征图上的点是如何传播到原图上的，如何计算特征图上的点到原图的感受野(Receptive Field)？</strong></span>

这里就涉及到**感受野**的概念以及相关的计算方式了。所谓的感受野，是指经过卷积操作后，卷积特征图上的一个点，在前面的层（最前面就是原始图片）对于区域的大小。

在这里，我并不打算细讲这个问题，因为一来计算方式有些繁琐，而来已经有不错的参考文献了。详细的计算方式可以参考引用
[[4](https://zhuanlan.zhihu.com/p/24780433)]
[[5](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)]。我简单的说下计算方式：

* **特征图上的点是如何传播到原图上的：**
前面我们提到，卷积网络中的卷积核都是$$[size=3 \times 3, padding=1, stride=1]$$ 的规格，这导致卷积操作前后特征图的大小并不会发生变化，整个网络是通过 pooling 层来进行下采样的。由于共有4个pooling层，因此坐标的映射关系可以简单的认为是 $$2^4 = 16$$ 的线性关系。也就是说如果特征图上的点是$$(3, 4)$$，那么其感受野的中心点在原图上的坐标应该是$$(48, 64)$$。

* **特征图上一点的在原图上的感受野是多少：** 在卷积操作的过程中，假设有如下的符号定义：
    * 卷积核大小为 $$k \times k$$;
    * 卷积的步长 stride 定义为 $$s$$;
    * 卷积过程中的 padding 定义为 $$p$$;
    * 卷积过程输入图的边长定义为 $$n_{in}$$, 输出图的边长定义为 $$n_{out}$$；

我们都知道，输入输出关系图的映射公式为：$$n_{out} = ⌊\frac{n_{in} + 2p - k]}{s}⌋ + 1$$。那么我们逆向思维可以想到，输出图中的一点，对应上一层的影响范围是多大呢？$$n_{in} = (n_{out} - 1)*s + k$$（注意的是，padding不应该考虑进去）。基于这种推理方式，我们可以得到基于VGG16的 Faster-RCNN中，特征图 `conv5_3`中一点的感受野范围是多少：


<figure>
	<a href=""><img src="/assets/images/faster-rcnn/rf.png" alt="" width="240" heigh="400"></a>
    <figcaption><a href="" title="">`conv5_3`在每层的感受野</a></figcaption>
</figure>

基于上面的两个子问题，我们可以计算出，在 `conv5_3` 上 $$3 \times 3$$卷积窗口，在原图中的感受野为$$(196 + 16*2) \times (196 + 16*2)$$


<span style="color:#1A2F7B;font-size:120%"><strong>3. 这9个Anchor-boxes的大小都超出了这 3x3 卷积的感受野，检测还能有效吗？</strong></span>

这个是一个很犀利的问题，我曾经为这个问题苦苦寻找了很久的答案。我曾经无法理解 $$3 \times 3$$ 卷积窗口的结果$$512$$维特征和这9个anchor-boxes的关系。这个问题在知乎上也有广泛的讨论：
* [知乎：faster rcnn中rpn的anchor，sliding windows，proposals？](https://www.zhihu.com/question/42205480)
* [知乎：为什么faster rcnn中特征图的感知野会小于某些anchor的大小？](https://www.zhihu.com/question/264964580)

经过多个答主和自己的思考，我认为有些高票回答是**错误**的。其中这些错误的观点有：

1. Anchor本质是 SPP 思想的逆向。其指出这$$3 \times 3$$的滑动窗口，是原始图片中9个anchors的作用结果。
2. 答非所问，没有正面回答$$3 \times 3$$卷积和9个anchor-box的关系，只是大量的图片说明 anchor-box 是如何生成的。

为什么说上面的答案是**错误**的呢？因为我们知道anchor在原图上的尺寸各异($$\{128^2, 256^2, 512^2\}$$)，而$$3 \times 3$$滑动窗口的感受野范围也才不过$$228^2$$，**所以anchor是超出了感受野的范围的。也就是说，$$3 \times 3$$的卷积窗口，根本很可能只蕴含了某些anchor-box的部分信息！**

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/rf2.png" alt="" width="300" heigh="400"></a>
</figure>

所以我觉得 RPN 网络中，这个$$3x3$$的网络具有两个作用：

1. 这是一种类似**『窥一斑而见全豹』**的思路，感受野中有可能会看见了整个物体，也有可能只看见了物体的一部分（车身），从这个**部分信息**出发，通过不同尺度的检测框来预测这是不是一个物体，以及预测车身和车头的位置大概在哪。
2. 这个$$3\times 3$$的卷积层，后接了一个`RELU`操作，保持了整个`conv5_3` 的feat-map 大小不变的前提下，增加了一些非线性的变化，一次来**提高后面的分类和回归任务的表达能力。**
3. 之所以不使用如 $$5 \times 5$$的滑动窗口，可能因为 $$ 5 \times 5$$ 卷积操作如果要保证输入输出特征图一致的话，就必须 $$padding=2$$，这样引入的噪声可能会比较多。

### RPN网络中的分类与回归任务

在仔细介绍分类和回归任务之前，我们先从模型的训练数据、结果输出、损失函数等全局的细节谈起。

#### **1. 训练样本的生成：**
如果输入大小为$$1000 \times 600$$，那么特征图 `conv5_3` 的大小大约为 $$60 \times 40$$，那么一共会生成约 $$20k$$ 个anchor-boxes。显然这数据实在是太多了，需要对其进行一定程度的筛选：

1. 如果该anchors-box超出了图片边界，过滤之；这大概可以剩下6000个anchor-boxes；论文中指出如果不去掉这些边框，会引入很多的额外 error-term，导致模型难以收敛。
2. 对每个标定的ground truth，与其重叠比例IoU最大的anchor记为正样本，这样可以保证每个ground truth至少对应一个正样本anchor;
3. 对每个anchors，如果其与某个ground truth的重叠比例IoU大于0.7，则记为正样本(目标)；如果小于0.3，则记为负样本(背景);
4. 再从已经得到的正负样本中随机选取256个anchors组成一个minibatch用于训练，而且正负样本的比例为1:1,；如果正样本不够，则补充一些负样本以满足256个anchors用于训练，反之亦然。

#### **2. 损失函数：**

由于涉及到分类和回归，所以需要定义一个多任务损失函数(Multi-task Loss Function)，包括Softmax Classification Loss和Bounding Box Regression Loss，公式定义如下：

$$
L(\lbrace p_i \rbrace, \lbrace t_i \rbrace)=\dfrac{1}{N_{cls}}\Sigma_i L_{cls}(p_i,p_i^{\ast}) + \lambda \dfrac{1}{N_{reg}}\Sigma_i p_i^{\ast} L_{reg}(t_i, t_i^{\ast})
$$

其中，分类部分，损失函数用的就是交叉熵损失函数了，其中的$$p_i$$是样本分类的概率值，$$p_i^*$$为样本的标定值，如果anchor包含物体，则该值为1，否则为零。而 bbox-regression 部分，具体可以参看本博客中在 [RCNN 博文](http://http://dreamingo.github.io/2018/07/rcnn/)中介绍的 bbox-regression 算法。这里不再展开介绍。

#### **3. 预测过程中模型输出：**

Region-Proposal 是图像输入到 RPN 网络中的进行一次 forward 计算的输出：

1. 计算特征图 `conv5_3` 映射到到图像的所有 anchor-box（20k个），通过RPN网络计算出这些anchor-box的 bbox 回归偏移向量 和 分类score（包含object的概率）；
2. 通过anchors的坐标和偏移向量，计算得到实际预测框proposals的坐标信息；
3. 对超出边界的proposal，对其进行 clip 处理，并过滤掉长宽小于阈值的proposal；
4. 对剩下的proposal按照分类得分（fore-ground score）进行排序，并选择出约前6000个proposals；
5. 执行NMS（非最大值抑制算法），在根据NMS后的 foreground-score 排序，输出前300个作为输出。

#### **4. 9个分类器和回归器：**

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/cls.png" alt="" width="600" heigh="200"></a>
    <figcaption><a href="" title="">RPN网络中的分类和回归任务架构</a></figcaption>
</figure>

从上图中我们可以看到，对于分类任务，模型利用 $$1\times 1 \times 18$$的卷积层来对特征进行整合后输入到 Softmax 层，对于每个$$3 \times 3$$滑窗的512维向量，该层相当于同时训练了**9个二分类器**，共有18个输出结果。

> 为了便于Softmax分类，需要对分类层执行reshape操作，这也是由底层数据结构决定的。在caffe中，Blob的数据存储形式为Blob=[batch_size,channel,height,width]，而对于分类层(cls)，其在Blob中的实际存储形式为[1,2k,H,W]，而Softmax针对每个anchor进行二分类，所以需要在分类层后面增加一个reshape layer，将数据组织形式变换为[1,2,k∗H,W]，之后再reshape回原来的结构，caffe中有对`softmax_loss_layer.cpp`的reshape函数做如下解释：

```
"Number of labels must match number of predictions; "  
"e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "  
"label count (number of labels) must be N*H*W, "  
"with integer values in {0, 1, ..., C-1}.";
```

而对于回归任务，因为每个 anchor-box 需要预测4个偏移量，因此，该回归任务也相当于同时训练了**9个回归器（针对不同的尺寸的anchor-box有一套自己的参数）**，共输出36个结果。


## 0x05. Fast-RCNN 模型

RPN 网络生成了一些列的 region-proposals 后，就需要输入到 Fast-RCNN 网络中，针对每个 region-proposal 做进一步的物体分类和 bbox 坐标回归。Fast-RCNN 是在 RCNN 的痛点上提出的，我们来再简单复述下其想解决的问题：

> 1. Fast-RCNN 论文指出，RCNN预测过程之所以慢，是因为其需要对2000多个 region-proposal 区域进行**重复的特征提取操作**。其通过 **`RoI-Pooling-Layer`** 解决了这个只对原始图片进行一次特征提取的过程，将每个region-proposal 映射到 feature-map 上对应的特征块，从而实现同一张图中不同 proposal **共享计算**的问题，大大的加快了算法过程。
> 2. 不再把分类 和 bbox 回归问题分为两个阶段，将在一个网络中同时完成 分类 和 bbox 回归任务。

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/fast.png" alt="" width="700" heigh="300"></a>
    <figcaption><a href="" title="">红框处表示 Fast-RCNN 的网络架构</a></figcaption>
</figure>

### 1. RoI Pooling

RoI(Region of Interest) Pooling Layer 的提出是基于以下的这个问题的：

**Fast-RCNN 最后连接的全连接层是对输入特征图的大小规格有限定的。不同 proposals 大小不一样， 在feature-map 上映射得到区域也不一样，那么该如何统一输出尺寸呢？**

因此，从上面的一段话中，我们可以简单的得出了 `RoI-Pooling-Layer` 的两个功能：

> 1. 将原图中的 proposals 区域映射到特征图 `conv5_3` 上对应的区域(patch);
> 2. 将该区域(patch)的处理，输出统一的尺寸。

对于第一个功能点，前面已经提到过如何将原始图片和特征图上的点坐标作相互之间的映射。由于基于VGG16的Fast-RCNN模型在`conv5_3`的 $$\{strides=2^4=16\}$$，因此，我们可以看到在代码 `faster-rcnn/py-faster-rcnn/caffe-fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp`有相关的区域映射代码：

```cpp
...
// 其中 spatial_scale_ 是 RoI 层的定义参数，在这里该值为 1/16
int roi_start_w = round(bottom_rois[1] * spatial_scale_);
int roi_start_h = round(bottom_rois[2] * spatial_scale_);
int roi_end_w = round(bottom_rois[3] * spatial_scale_);
int roi_end_h = round(bottom_rois[4] * spatial_scale_);
CHECK_GE(roi_batch_ind, 0);
CHECK_LT(roi_batch_ind, batch_size);

int roi_height = max(roi_end_h - roi_start_h + 1, 1);
int roi_width = max(roi_end_w - roi_start_w + 1, 1);
..
```

对于第二个问题，ROI-Layer采取以下的办法：

1. 将每个proposal映射到 `conv5_3` 中的特征图的一个区域patch(大小为$$ H' \times W' \times C $$)；
2. 该 patch 分成 $$N \times N$$ 份（论文中$$N = 7$$）
3. 对这$$7 \times 7$$份，每份执行 Max-Pooling 操作；
4. 无论原始的proposal多大，其patch都会被映射为 $$ 7 \times 7 \times C$$ 固定尺寸的特征图。

<figure>
	<a href=""><img src="/assets/images/faster-rcnn/roi.png" alt="" width="400" heigh="400"></a>
    <figcaption><a href="" title="">RoI-Pooling 处理流程</a></figcaption>
</figure>

这里就贴出 RoI-Layer 是如何将patch切割成 $$7 \times 7$$ 份的源码的，大家可以参考下：

```cpp
// pooled_height_ & pooled_width_ 在这里都是等于7
// bin_size_w/h 表示分成7x7份，每份的大小。注意这里是Dtype=float类型。
const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height_);
const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width_);

const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

for (int c = 0; c < channels_; ++c) {
  for (int ph = 0; ph < pooled_height_; ++ph) {
    for (int pw = 0; pw < pooled_width_; ++pw) {
      // Compute pooling region for this output unit:
      //  start (included) = floor(ph * roi_height / pooled_height_)
      //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
      int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                          * bin_size_h));
      int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                          * bin_size_w));
      int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                       * bin_size_h));
      int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                       * bin_size_w));

      hstart = min(max(hstart + roi_start_h, 0), height_);
      hend = min(max(hend + roi_start_h, 0), height_);
      wstart = min(max(wstart + roi_start_w, 0), width_);
      wend = min(max(wend + roi_start_w, 0), width_);

      bool is_empty = (hend <= hstart) || (wend <= wstart);

      const int pool_index = ph * pooled_width_ + pw;
      if (is_empty) {
        top_data[pool_index] = 0;
        argmax_data[pool_index] = -1;
      }
```

### 2. 杂谈

RoI Pooling层后接多个全连接层，最后为两个子连接层——分类层(cls)和回归层(reg)，和RPN的输出类似，只不过输出向量的维数不一样。如果类别数为N+1(包括背景)，分类层的向量维数为N+1，回归层的向量维数则为4(N+1)。还有一个关键问题是RPN网络输出的proposal如何组织成Fast R-CNN的训练样本：

* 对每个proposal，计算其与所有ground truth的重叠比例IoU
* 筛选出与每个proposal重叠比例最大的ground truth
* 如果proposal的最大IoU大于0.5则为目标(前景)，标签值(label)为对应ground truth的目标分类；如果IoU小于0.5且大于0.1则为背景，标签值为0
* 从2张图像中随机选取128个proposals组成一个minibatch，前景和背景的比例为1:3
* 计算样本proposal与对应ground truth的回归参数作为标定值，并且将回归参数从(4,)拓展为(4(N+1),)，只有对应类的标定值才为非0。
* 设定训练样本的回归权值，权值同样为4(N+1)维，且只有样本对应标签类的权值才为非0。
* 在源码实现中，用于训练Fast R-CNN的Proposal除了RPN网络生成的，还有图像的ground truth，这两者归并到一起，然后通过筛选组成minibatch用于迭代训练。Fast R-CNN的损失函数也与RPN类似，二分类变成了多分类，背景同样不参与回归损失计算，且只考虑proposal预测为标签类的回归损失。

## 0x06. Fast-RCNN 模型与RPN网络的一体化

对于提取proposals的RPN，以及分类回归的Fast R-CNN，如何将这两个网络嵌入到同一个网络结构中，训练一个共享卷积层参数的多任务(Multi-task)网络模型。源码中有实现交替训练(Alternating training)和端到端训练(end-to-end)两种方式，这里介绍交替训练的方法。

* 训练RPN网络，用ImageNet模型M0初始化，训练得到模型M1
* 利用第一步训练的RPN网络模型M1，生成Proposal P1
* 使用上一步生成的Proposal，训练Fast R-CNN网络，同样用ImageNet模型初始化，训练得到模型M2
* 训练RPN网络，用Fast R-CNN网络M2初始化，且固定卷积层参数，只微调RPN网络独有的层，训练得到模型M3
* 利用上一步训练的RPN网络模型M3，生成Proposal P2
* 训练Fast R-CNN网络，用RPN网络模型M3初始化，且卷积层参数和RPN参数不变，只微调Fast R-CNN独有的网络层，得到最终模型M4
* 由训练流程可知，第4步训练RPN网络和第6步训练Fast R-CNN网络实现了卷积层参数共享。总体上看，训练过程只循环了2次，但每一步训练(M1，M2，M3，M4)都迭代了多次(e.g. 80k，60k)。对于固定卷积层参数，只需将学习率(learning rate)设置为0即可。

## 0x07. 总结

Faster-RCNN 工作作为 RCNN 系列的巅峰之作，无论是从检测精度和速度上来看都是棒棒的。这里面很多设计的思想（PRN，RoI等）都深刻着影响着后来工作。其优点我就不再阐述了（毕竟所有的亮点都是它的优点。）我再简单的说说其中一个缺点吧；

Faster-RCNN号称是一个一体化的系统工程（与RCNN相比那当然是了），并且通过一些取巧的办法吧 RPN 网络和Fast-RCNN网络糅合在一起。但是本质上，这还是一个 2-stage 的工作（先提出 proposal，再分类）。那能否再进一步一体化变成 1-stage 的工作呢？这就是后面 YOLO 等系列的工作咯~

## 0x08. 引用

1. [1][Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
2. [2][Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
3. [3][Faster R-CNN论文及源码解读](https://senitco.github.io/2017/09/02/faster-rcnn/)
4. [4][知乎专栏：原始图片中的RoI如何映射到到feature map?](https://zhuanlan.zhihu.com/p/24780433)
5. [5][A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)
