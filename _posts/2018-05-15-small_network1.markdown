---
layout: post
title: "卷积神经网络-快而小的网络总结：SqueezeNet & MobileNet"
modified:
categories: DeepLearning
description: "卷积神经网络-快而小的网络总结：SqueezeNet & MobileNet"
tags: [cnn deeplearning model_compress]
comments: true
mathjax: true
share:
date: 2018-05-15T13:54:50+08:00
---

<!--- {% include toc.html %} --->
<!--- {:toc} --->

本文主要结合 Reference 中的资料，介绍、总结和梳理最近两年出现的高效小网络。本文主要介绍早期的两个网络：
[SqueezeNet](https://arxiv.org/abs/1602.07360) & [MobileNet V1](https://arxiv.org/abs/1704.04861)

## Reference
+ [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and \< 0.5 MB model size](https://arxiv.org/abs/1602.07360)
+ [Mobilenets: Efficient convolutional neural networks for mobile vision applications](https://arxiv.org/abs/1704.04861)
+ [知乎：CVPR 2018 高效小网络探密（上）](https://zhuanlan.zhihu.com/p/37074222)
+ [知乎：卷积神经网络中用1\*1 卷积有什么作用或者好处呢？](https://www.zhihu.com/question/56024942)

## 背景介绍
2014，2015年前后，业界便开始探索如何针对大型的神经网络进行压缩和加速。主要的代表工作有参数裁剪、量化、知识蒸馏以及
卷积核矩阵低秩分解等途径。但是后来随着[GoogLeNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)，[NIN](https://arxiv.org/abs/1312.4400),
等网络中1x1卷积核的兴起，设计小而快的网络慢慢在这两年成为了减少网络参数、计算量方面成为了重要的角色。

下面先简单介绍下本文的两个主角 SqueezeNet 和 MobileNet 的主要情况：
* SqueezeNet: 项目源码：[DeepScale/SqueezeNet](https://github.com/DeepScale/SqueezeNet), berkeley和stanford合作论文，2016年2月挂arXiv，最新v4，ICLR 2017被拒。
注意的是，github项目中放出了 SqueezeNet_v1.1 的实现，**在计算量方面减少了2.4倍**；
* MobileNet: 项目源码：[tensorflow/models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md), Google论文，2017年4月挂arXiv，未投。
其中有趣的一点[参考回答[Alan Huang：如何评价mobilenet v2)](https://www.zhihu.com/question/265709710/answer/298245276)]是，MobileNet 这种复古的直筒结构，其实是Google内部两年前的工作，因为发现一直没人占这个坑，所以就挂到了arxiv上了...

## 1 X 1卷积核

在简单介绍这两个网络前，先简单的对1x1卷积核的作用唠叨一下。1x1卷积网络据我所知，
最初被提出的是在[Network in Nework](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1312.4400v3.pdf), 然后在Inception和ResNet中大放异彩。
那么，究竟如何直观的理解1x1卷积核的作用呢？

如果卷积的输入是一个平面，那么1x1卷积可能并没有什么作用，因为其完全不考虑像素与周边其他像素的关系。但是由于卷积的输入是一个长方体，所以1x1卷积
实际上是对每个像素点，在**不同的channels上进行线性组合**（起到信息整合的作用）,并且保留原有feature-map的平面结构，调整了depth，起到了完成升维和
降维的特点。在InceptionNet中，1x1的使用结构图如下：

![img](https://pic4.zhimg.com/80/v2-0f4430c2c6b79df077ffbe6957f0a99f_hd.jpg)

可以看到在下图，输入到3x3或者5x5卷积之前，都先通过一层1x1卷积来降低输入的维度，减少计算量；

所以总结而言，1x1卷积的主要作用有：
* 升/降维：例如输入特征图的channel数量为500，通过一个(500x1x1x30)的1x1卷基层后，输出特征图的
channel数目为30;
* 信息整合：上面提到，1x1是在不同的channels上进行线性整合，起到汇总信息的作用。在mobilenet中，
point-wise layer的作用便是如此；
* 假如非线性：因为每个1x1层后一般会再接一个RELU，提升网络的表达能力；

## SqueezeNet

SqueezeNet通篇的一个主旨是，保证模型精度的同时，必须尽可能的小。但是论文通篇并没有提到是否加快的实际的inference速度。
因此，SqueezeNet的一个设计原则是：**模型可以不快，但是必须尽可能的小**

文章在一开端的时候先是介绍了一通小模型的各种好处，例如：
* 分布式训练的时候，小模型可以令server之间的通信变少。
* 从云端到用户端更新模型的时候，更加的便捷和更少的bandwidth；
* 在FPGA或者别的嵌入式设备中更具优势；

论文提出了三个设计策略，这三个策略使得网络尽可能小的同时，提高模型精度；
1. **用1X1卷积代替3x3卷积**，同样的channel数目和卷积核情况下，参数数量减少9倍；
2. 3x3还是起到提取特征的作用，但是尽可能的**减少输入到3x3卷积层的输入特征图的channel数量**；继续减少3x3层的计算量和参数数量；
3. **延迟下采样**。所谓的延迟下采样，是因为如果特征图的分辨率更大，同等参数的情况下效果会更好。这squeezenet中，基本不采取pooling层
来进行下采样，卷积层采取 stride 1， padding 1的策略，使得卷积后feature-map的大小不变，提升精度。（squeezenet中通过stride 2来进行
下采样，这种延迟下采样的方法，**参数上没有任何的增加/减少，提高了中间层feature-map的分辨率从而提高的模型精度，但是比起传统的卷积层，
因为没有下采样，输入的feature-map变大，计算量也会增加**。）

### Fire-Module

SqueezeNet中提出了一个独特的卷积层模块-FireModule，该模块主要有以下组成：

1. Squeeze-Layer：压缩层，通过一个1x1的卷积核来对输入的feature-map进行降维，符合上面的设计原理2；
2. Expand-Layer：扩张层，主要由1x1卷积和3x3卷积混合而成，由于3x3卷积层stride，padding为1，所以输出的特征图大小不变，因此，Expand-layer最后将1x1和3x3的卷积输出concat在一起，作为整个fire-module的输出；
<figure>
	<a href=""><img src="/assets/images/small_model1/fire_module.png" alt="" width="500" heigh="300"></a>
</figure>

#### 计算量和参数数量分析

在分析之前，我们先定义一系列的参数符号：
* 输入到 fire-module 的特征图大小为 $$H \times W \times C $$
* Squeeze层1x1卷积核的数目为 $$s_{1 \times 1}$$
* Expand层的1x1卷积核和3x3卷积核的数量分别是$$e_{1 \times 1}, e_{3 \times 3}$$

那么，一个Fire-module的参数数量为：$$1 \times 1 \times C \times s_{1\times 1} + 1 \times 1 \times s_{ 1 \times 1} \times e_{1 \times 1} + 3 \times 3 \times s_{1 \times 1} \times e_{3\times 3} = S_{1\times 1}(C + e_{1\times 1} + 9e_{3\times 3})$$ \_

如果正常的3x3卷积层，保持同样的输出和输入大小，则参数数量是 $$9C(e_{1\times 1} + e_{3 \times 3})$$。按照SqueezeNet中fire7的结构（$$C=384，s_{1\times1} = 48, e_{1\times 1} = e_{3 \times 3} = 192_{}$$），则前后压缩率达到12倍；**同理于计算量的减少** 

> SqueezeNet共8个Fire Module，2个CONV和4个POOL，没有BN，最终模型4.8M，在ImageNet上top-1 acc 57.5%, top-5 acc 80.3%，**性能是AlexNet水平**，经过DeepCompression进一步压缩后模型大小可以达到逆天的0.47M，但DeepCompression方法也是仅关心压缩不关心加速的。最后实验还测试了shotcut，证明类似ResNet那样最简单的shotcut最好，top-1和top-5分别提升2.9%和2.2%，性能提升明显且不会增加参数数量，几乎不会影响计算量，shotcut简直是a free lunch！ -- by [CVPR2018 高效小网络分析](https://zhuanlan.zhihu.com/p/37074222)

![img](https://pic3.zhimg.com/80/v2-71c07aaf3ebc71011e9defb908574e77_hd.jpg)

第一个CONV1x1将输入通道数压缩(squeeze)到1/8送给CONV3x3，上下两路CONV扩展(expand)四倍后联结，还原输入通道数量。block中CONV1x1和CONV3x3的参数数量之比是1:3，计算量之比也是1:3。

SqueezeNet中虽然大量使用CONV1x1，但CONV3x3占比依然在75%以上，inferece framework中CONV3x3的优化对速度影响更大。

### 杂谈

* SqueezeNet在conv10层，为了节省最后一层全连接层，直接在输入feature-map为(13x13x512)的情况下，接了一个1x1x1000的卷积层。这样子该层的参数从全连接层（13x13x512x1000 = 8652.8w）变成了conv10的参数数量(1000 x 512),直接减少了169倍的参数。但是却导致计算量也是8652.8w，这是非常低效的。（例如在mobilenet中，到FC层前会先做一次Global Average Pooling，使得计算量也就只有1M左右。）

* 最新推出的[SqueezeNetV1.1](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1),主要是改进了第一层卷积层的kernel大小，以及将pooling层提前了。这使得网络比原来少了2.4倍计算量，但是大小和精度却基本没有变化；

<figure>
	<a href=""><img src="/assets/images/small_model1/squeezenet1_1.png" alt="" width="500" heigh="300"></a>
</figure>

* 在次科普一个有用的tools[Netscope CNN Analyzer](http://dgschwend.github.io/netscope/quickstart.html)，只要上传对应的Caffe's prototxt fommat文件，能够自动将网络结构、每层参数量，计算量计算出来。

- - -

## MobileNet

MobileNet是第一个面向移动端的小网络，其论文中也指出目前很多小网络的设计更加focus在模型的大小压缩上，而并没有在预测速度上做出比较大的贡献。而Mobilenet的设计兼顾小模型和速度快的原则。在MobileNet论文中，关键在于模块**Depthwise Sparable Convolution**深度分离卷积的设计。

在介绍Deptchwise Sparable 卷积之前，打算先介绍一下同样原理下，更通用的 **Group-Convolution 模块**;

**GCONV分组(Group)卷积层**，在该卷积层之前，可能会现有一个1x1的卷积层做输入的升降维。1x1卷积的输出按通道数划分为 g 组，每小组独立分别卷积，结果联结到一起作为输出，如图中上面部分，Output channels作为GCONV的输入，分了3组分别CONV3x3，组内有信息流通，但不同分组之间没有信息流通。

![img](https://pic2.zhimg.com/80/v2-50e1dd66d896408716881659b10be119_hd.jpg)

* **参数数量：**  $$(k_h \times k_w \times c_{in}/g \times c_{out}/g) \times g$$
* **计算量（mAdds）：** $$(k_h \times k_w \times c_{in}/g \times c_{out}/g) \times g \times H \times W $$

而DWCONV（depthwise-convolution)层，则是 GCONV 的极端情况，当分组的数目 g 等于输出的通道数目，即$$g = c_{in}$$，每个卷积负责对特征图的每一维来提取特征；_

我们都知道传统的卷积操作可用kernel来抽取特征并且按照通道将其线性组合成新的特征。因此，mobilenet 中的主要思想便是将上面的操作分为两步：
* **depth-wise convolution:** 对每个input-channel执行一个卷积操作，一般是一个3x3的卷积核；
* **point-wise convolution:** 1x1用于对depth-wise convolution 抽取得到的特征做线性组合。

<figure>
	<a href=""><img src="/assets/images/small_model1/mobile_net.png" alt="" width="350" heigh="500"></a>
</figure>

### 复杂度分析

在进行复杂度分析之前，我们先定义以下符号：
* 卷积核大小为 $$D_k \times D_k $$
* 输入的特征图大小为$$H \times W \times M$$, 其中 M 是特征图的通道数目；
* 卷积层中共有 N 个卷积核；

因此，参数复杂度和计算复杂度如下（计算复杂度还需乘以输出特征图的大小$$H \times W$$）
* **传统卷积参数数量：** $$ D_k \times D_k \times M \times N $$
* **Depth-wise Speraable参数数量：** $$D_k \times D_k \times M + 1 \times 1 \times M \times N $$

因此，DCONV结构的参数数量(可以推出计算量也是一样的)是原来标准卷积层的$$\frac{1}{N} + \frac{1}{D_k^2}$$，
如果$$D_k = 3$$的话，那么计算量就仅仅比原来少了8到9倍左右。其中值得注意的一点是，同一层中depth-wise convoluton
和 point-wise convolution参数和计算量比例是$$\frac{D_k^2}{N} $$，在mobilenet中，这个比例约等于几十倍；，**因此，
在mobilenet 中， 1x1卷积的计算量占据了极大的比例，如何有效的实现1x1的卷积算法，是加快mobilenet的核心关键。**

<figure>
	<a href=""><img src="/assets/images/small_model1/mobile_net2.png" alt="" width="300" heigh="400"></a>
</figure>

### 网络结构
* 每个卷积层后面，都会接上一个 BN 层和 REUL 层；
* 下采样是通过depthwise-layer的步长为2实现的；
* 在最后的全连接层之前，会有一个gobal-average pooling层，将7x7x1024的特征图变成1x1x1024;
* 传统的卷积操作都是通过高度优化的 general-matrix-multiply-functions(GEMM)来计算的，但是在这之前往往会有一个`im2col`的内存
操作将数据进行重排；但是1x1卷积不需要这个操作。可以直接通过GEMM函数高效实现；
* 在训练的过程中，重要的一点是，对于depth-wise layer，尽可能的用较小的 weight-decay（l2 regularization），
因为这个卷积核本来参数就少，不会因此overfitting问题；

### Width Multipiler & Resolution Multiplier

论文中提出了两个乘子 $$\alpha$$ 和 $$\rho$$，其中：
* **宽度乘子$$\alpha$$**控制了网络的深度。这个是通过控制每层1x1卷积的数目（降维的目的），
因为1x1卷积核的输出变少了，因此到下一层的输入也变少了。粗略估计，widht-multipiler能够大概的减少参数和计算量$$\alpha^{2}$$倍；
* **分辨率乘子$$\rho$$**更是粗暴，直接控制输入图片分辨率的大小，从而控制了每层的H和W，达到减少计算量的目的，减少的量大概也是$$\rho^{2}$$；

$$
D_k \times D_k \times \alpha M \times \rho^{2}D_F^{2} + \alpha M \times \alpha N \times \rho^{2} D_F^{2}
$$

- - -

## 总结&杂谈

### 实际预测速度分析
在腾讯的[ncnn框架的benchmark](https://github.com/Tencent/ncnn/tree/master/benchmark)，在高通和海思等硬件实测测试，
发现SqueezeNet反而比MobileNet要更快一些。唯一的猜测是这里SqueezeNet采取的1.1版本，利用工具[Netscope CNN Analyzer](http://dgschwend.github.io/netscope/quickstart.html)
做分析，可以得出这三个网络的理论计算复杂度。因此SqueezeNet更快一点也是可以理解的。不过，SqueezeNet是AlexNet级的精度，如果真的要和
MobileNet比，应该是和MobileNet差不多精度的来比（0.5MobileNet-160，也就是$$\alpha=0.5, \rho=0.71$$），后者仅有38M mAdds;

* SqueezeNetV1.1：387.75M mAdds
* MobileNet1.0： 573.78M mAdds
* MobileNet2.0: 438.04M mAdds

### 1x1卷积Caffe实现
上面有提到，1x1卷积不需要执行`im2col`操作，果然，在Caffe源码中，就找到这么一段：
{% highlight c++ %}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}
{% endhighlight %}
