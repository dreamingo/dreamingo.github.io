---
layout: post
title: "Transformer系列：浅读 Atttention is All You Need 论文"
modified:
categories: 机器学习 
description: "deep-learning transformer attention"
tags: [deep-learning transformer attention]
comments: true
mathjax: true
share:
date: 2019-04-28T11:08:38+21:12
---

> 本文是 Transformer 系列的第一篇文章，旨在对经典论文 Atttention is All You Need 提出一些简单的见解，对 Transformer 架构 初窥门径。

## 0x01. 引言
Transformer 是2017年Google 经典论文 Atttention is All You Need 中提出用于『颠覆』RNN 结构的新框架，在这些年来引起了巨大的关注和反响，特别是在2018年 BERT 的横空出世，更是引起了
大家对 Transformer 结构的重视，同时也成为了每位 NLPer 必读的论文之一。

在动笔之前，我一度非常犹豫和挣扎，因为如今网上已有众多优秀的资料阐述 Transformer 的细节，文章如何写得不太同质化，有自己独特的见解是有一定困难的。思来想去，还是决定按照自己的学习脉络
和理解来**阐述、总结和升华**  Transformer 的技术细节，算是加强自身的理解和表述能力。同时我会将所有参考到的资料引用下方，方便大家查看。

在读完 RNN 系列文章后，相信大家对 Encoder-Dedocer 和 Attention 机制有一定的了解。在开始阅读本文章之前，
同样强烈的建议大家阅读这篇经典的Transformer 入门博客：[Jay Alammar - The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)


与 Seq2Seq 结构类似，Transformer 中同样也包含了 Encoder 和 Decoder 组件。其中，encoding 模块是由 6 层 encoder-layer 堆叠起来的，decoding 模块同样也是由 6 层 decoder-layer 堆叠起来。

<figure>
	<a href=""><img src="/assets/images/transformer/transformer.jpg" alt="" width="500" heigh="500"></a>
    <figcaption><a href="" title="">Transformer的 Encoder-Decoder 堆叠结构</a></figcaption>
</figure>

- - -

## 0x02. Encoder 架构
如下图，Transformer 的Encoder Layer 包含两层的Sublayer，分别是 Self-Attention Layer 和 Feed-Forward-Neural-Network。
<figure>
	<a href=""><img src="/assets/images/transformer/encoder.jpg" alt="" width="700" heigh="500"></a>
    <figcaption><a href="" title="">Transformer的 Encoder-Layer 结构</a></figcaption>
</figure>
其中，Encoder层的**输入**，对于第一层，自然就是每个单词的embedding vector；对于每一个输入vecotr $$x_i$$，都会**输出**另一个向量 $$r_i$$，作为下一层的**输入**。

如果用矩阵的方式来表示，假设输入矩阵 $$\mathbf{X} \in R^{n \times d_k}$$，其中 $$n$$ 表示输入序列的长度。而 $$d_k$$ 则表示输入序列中每个位置的向量长度。则Encoder Layer
的输出 $$\mathbf{R} \in R^{n \times d_k}$$。**这相当于将原来 $$n \times d_k $$ 的序列 $$Q$$ 经过 encoding 层之后重新编码成一个新的 $$n \times d_k $$ 序列；**

### 2.1 Attention 结构：

#### **2.1.1 剧情回顾：**

在开始介绍Self-Attention 这一『新概念』 之前，我们先复习下之前 Seq2Seq 中的 Attention 概念：
在 Seq2Seq 的decoder阶段，每一步 timestep $$t$$ 的输出，需要结合**输入**的 Encoder 阶段的 hidden-states $$\{\bar{h_1}, \bar{h_2}, .. \bar{h_n}\}$$和当前decoder产生的hidden-state $$h_t$$，
来**输出**该步骤的 context-vector $$c_t$$。在具体的代码实现时，Attention层的输入输出如下：

{% highlight python %}
def call(self, query, values):
    """ 每调用一次，生成一个timestep t 所需的attention信息：
    Args:
        query:  Tensor, with shape:[batch_sz, hidden_size],
            在这里其实就是$$t-1$$ 的hidden-state $$h_{t-1}$$
        values:  Tensor, with shape [batch_sz, max_input_len, hidden_size]
            在这里是Encoder阶段中生成的所有hidden-stat：
            {\bar{h}_1, ....\bar{h}_{T_x}}
    Returns:
        context_vector:  Tensor, with shape: [batch_sz, hidden_size]
        attention_weights:  Tensor, with shape: [batch_sz, max_input_len, 1]
            在该示例中，返回这个值主要是用户绘制attention明亮图；
    """
    ...
{% endhighlight %}

总而言之，Attention 层的作用，就是整合**该关注的数据**（例如Seq2Seq中输入序列的hidden-state、SelfAttention中自身序列的数据），**为每个位置输出生成** $$t$$ 的上下文向量 $$c_t$$

#### **2.1.2 Attention模式的全新定义：**

在 Attention is All You Need 一文中，其将Attention的形式抽象成以下的三要素： 
* **Query**:  $$Q \in R^{n \times d_k}$$, 其中 $$n$$ 为Query向量的集合大小，$$d_k$$表示Query向量的长度；
* **Key**:    $$K \in R^{m \times d_k}$$, 其中 $$m$$ 为Key 向量的集合大小，$$d_k$$表示Key向量的长度；
* **Value**:  $$V \in R^{m \times d_v}$$，其中$$m$$ 为Value向量的集合大小，$$d_v$$表示Value向量的长度；

同时定义Attention的计算方式：并且模型最终输出 $$n \times d_v$$ 的序列，**这相当于为序列的 $$n$$ 个位置生成其特有的context vector**

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

看完上面新的定义之后，可能大家会对新的这种架构术语一脸懵逼。Query、Key、Value分别表示什么呢？如何抽象的理解其中的意思？在这里我用一种比较**民科**的方式来抽象解释这三个术语所代表的含义：

> 假设我们处于一个搜索场景中。用户输入Query，我们需要针对该Query，只能返回一个综合的搜索结果（Context vecotr）；那么第一步，搜索引擎（AttentionLayer）会将 Query 和数据库中的 Key 计算**相似性**(点积计算相似性，Softmax归一化)，得出每份文档的权值比重。然后再根据计算得到的权值比重，加权糅合数据库中的 Value，输出一份最终的综合搜索结果(Context vecotr)

更进一步的，我们将这个新的定义模式套在 Seq2Seq 中的Attention模式，就会发现：

* Query: 其等于Decoder中每一步计算得到的隐含向量 $$h_t$$；
* Key: 其等于Encoder中产生的所有hiddent-states：$$\{\bar{h_1}, \bar{h_2}...\}$$
* Value = Key


### 2.2 Self-Attention & MultiHead：

在上面介绍完了论文中新提出的 Attention Layer 的形式，我们来看看 Self-Attention 层的组织形式。
顾名思义，Selft-Attention ，在序列内部做Attention，寻找序列内部的联系。具体而言是在将位置 $$t$$ 的输入向量编码成新的向量时，综合考虑输入序列**其他位置**的信息。例如：

在下图中，句子 "`The animal didn't cross the street because it was too tired`"，当我们encode单词`it`时，可以看到 self-attention
机制使"`it`"单词的 encoded 向量和"`the animal`" 产生了主要的关联：

<figure>
	<a href=""><img src="/assets/images/transformer/self_att_example.jpg" alt="" width="400" heigh="400"></a>
    <figcaption><a href="" title="">单词"it"的encoded向量，看到"the animal"贡献了很大的比例</a></figcaption>
</figure>

#### 2.2.1 **Self-Attention 层的输入是什么：**
以矩阵表示为例，每层Self-Attention Layer的输入矩阵$$X \in R^{d \times T_x}$$，其中 $$T_x$$表示输入序列的长度，而 $$d$$ 则表示编码向量的大小。以第一层为例子，则表示长度 $$T_x$$ 的输入序列，
每个位置用 embedding-size 为 $$d$$ 的向量表示。

#### 2.2.2 **$$Q$$, $$K$$, $$V$$矩阵具体如何获得：**

每个SelftAttention层中，有 $$W_Q \in R^{d \times d_q}$$, $$W_K \in R^{d \times d_k}$$，$$W_V \in R^{d \times d_v}$$ 三个kernel 矩阵，用于生成 $$Q$$, $$K$$, $$V$$ 矩阵数据：

<figure>
	<a href=""><img src="/assets/images/transformer/self_attention.jpg" alt="" width="600" heigh="400"></a>
    <figcaption><a href="" title=""> SelftAttention的矩阵具体表示形式：从输入X到输出Z </a></figcaption>
</figure>

#### 2.2.3 **什么是MultiHead**

在上述的第二步中，我们展示了Self Attention层中输入 $$X$$ 是如何 生成输出 $$Z$$ 的。假设定义Self Attention层中的kernel $$W = [W_Q, W_K, W_V]$$。
那么，所谓的MultiHead，其实就是：**定义多个kernel $$\{W_0, W_1, W_2...\}$$，多做几次 上述的 SelfAttention 过程，将生成的 $$\{Z_0, Z_1, Z_2..\}$$ concat 连接起来。**，然后再乘以一个权重矩阵 $$W_o$$，最终输出结果 $$Z$$

根据论文中的描述，Multi head 通过不同kernel 来做attention，使得模型在不同的子空间（subspace）对attention 进行建模。**MultiHead 的本质，其实和CNN 卷积kernel的概念非常类似。
通过不同的kernel抽取不同的特征，来达到从不同子空间对数据进行建模的效果。**


<figure>
	<a href=""><img src="/assets/images/transformer/multihead.jpg" alt="" width="600" heigh="600"></a>
    <figcaption><a href="" title=""> MultiHead 的具体形态 </a></figcaption>
</figure>

### 2.3 Position-wise Feed-Forward Networks

在 encoder 和 decoder 中，每个层的最后都包含一个全连接层（fully connected feed-forward network）；
在论文中，之所以加上 **positional-wise** 的描述，是因为经过Attention层后，每个位置的的输出 $$z_i$$
都会单独的过一次这个全连接层。每个位置相互之间是分离且独立的。但是值得注意的是，在同一个sublayer中，**不同的position其实共享的同一个kernel**；

$$
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
$$

其中，FFN 层的输入 $$X \in R^{d_{input} \times d_{model}}$$，可以理解为共有 $$d_{input}$$ 个输入，经过 Attention 层后，每个输入被编码成 $$d_{model}$$的向量；
些行进行相同的线性变换后，维度改变，重新堆叠成FFN层的输出。行与行之间无交错，完全是“separately and identically”，按位置进行变换

在论文中，作者将上述的两个全连接层类称为两个 $$1 \times 1$$ 的卷积层（全连接层本来就是可以等价于 $$1 \times 1$$的卷积层）
$$W_1 \in R^{d_{model} \times d_{ff} \times 1 \times 1}$$，而$$W_2 \in R^{d_{ff} \times d_{model} \times 1 \times 1}$$。
对于输入 $$X \in R^{1 \times d_{model} \times d_{input} \times 1}$$。$$1 \times 1$$ 卷积的作用相当于：

> 1. 对输入的数据通道进行信息整合，起到汇总消息的作用；
> 2. 根据自身的kernel数量，对输入数据起到升降纬得作用，并添加了非线性变化能力；

FFN 相当于将每个位置的Attention结果映射到一个更大维度的特征空间(从 $$d_{model}$$ 到 $$d_{ff}$$)，然后使用ReLU引入非线性进行筛选，最后恢复回原始维度。


## 0x03. Decoder 架构

Transformer论文中，Decoder模块同样是由6层相同的 decoder-layer 组成。 其中，
在经历 Encoder 阶段后，每个位置的单词都被重新编码成新的向量$$R_i$$，结合前面的Attention框架，其实 $$R$$ 就是对于decoder层中的Attention机制的 Key & Value。
每个Decoder Block 会结合本层得输入和来自encoder的key&value，对本层输入和输入句子进行attention建模；

<figure>
	<a href=""><img src="http://jalammar.github.io/images/t/transformer_decoding_1.gif" alt="" width="700" heigh="400"></a>
    <figcaption><a href="" title=""> Encoder 和 Decoder 层的具体联动</a></figcaption>
</figure>

在 Decoder 层中，和 encoder 不一样的是，有三个sublayer，其实多了中间的一个 encoder-decoder-attention 层。该层主要的作用是接收来自 encoder 的Key、Value数据，
和自身的输入Query数据进行 MultiHead Attention 交互。达到对输出句子和输入句子产生attention关联的效果。这点和 Seq2Seq 中的 attention 形式基本一致。

<figure>
	<a href=""><img src="/assets/images/transformer/decoder.jpg" alt="" width="400" heigh="400"></a>
    <figcaption><a href="" title=""> Decoder 的具体架构：三个sublayer</a></figcaption>
</figure>

其中，在Decoder阶段有几个值得注意的地方：

* **位置穿越问题：** 在Self-Attention Layer 中，位置 $$i$$ 不应该和 $$i+1, i+2...$$ 等后面的位置产生attention 关联，因为后面的数据还没被**预测**出来，不应该出现类似数据穿越的问题。
因此，在Decoder的SelftAttention过程中，在计算 $$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$，在计算softmax阶段是，通过添加一个mask矩阵，将位置 $$i$$ 
之后的数据都置为 $$-\inf$$(因为负无穷的softmax结果为0)。这种便捷的做法，既不影响整体的矩阵计算方式，通过简单的添加mask矩阵来达到这个小需求。

* **Teacher-Forcing:** 具体的Teacher-Forcing的技术可以参考上一篇的博文或者这个[链接](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)，
这种技术在 Encoder-Decoder 架构的模型训练中被广泛的使用。总结而言，就是在decoder训练结果，并不会拿上一次的输出 $$y_{t-1}$$ 作为下一次的输入，而是将grounp-truth $$\hat{y}_{t-1}$$
作为输入。这样避免了依赖前一阶段的输出，达到真正的并行训练。


## 0x04. 重要的Trick

### 4.1 Positional Embedding

结合上面的介绍信息，与RNN等序列模型相比，**Transformer具备了并行的优势，但是却丢失了捕捉序列顺序的能力！**
而对众多任务，特别是NLP任务而言，序列的顺序往往含有着重要的局部和全局信息。缺乏对序列顺序的捕捉能力的话，那么上述阐述的模型也仅仅是个精致的词袋（Bag of Words）模型。

因此，在 Transformer 论文中，提出了 Positional embedding 的概念。要对位置信息进行编码，那么该编码应该**包含绝对和相对位置的信息**，对此，论文中提出一种编码方式：
对于位置$$p$$，其 positional embedding 是一个 $$d_{pos}$$ 纬的向量。并且向量中的每一个元素 $$i$$ 的定义如下：

$$
\left\{\begin{aligned}&PE_{2i}(p)=\sin\Big(p/10000^{2i/{d_{pos}}}\Big)\\ 
&PE_{2i+1}(p)=\cos\Big(p/10000^{2i/{d_{pos}}}\Big) 
\end{aligned}\right.
$$

对于上面公式，之所以选择三角函数，最重要的一个原因是对**相对位置**进行建模：$$sin(a+b) = sinacosb + cosbsina$$, $$cos(a+b) = cosacosb - sinasinb$$，
这表明了位置 $$p + k$$ 的位置可以通过通过位置 $$p$$ 进行线性变化得到;

而上面的positional embedding是如何和模型输入结合的呢？模型的输入是句子的词向量信息，而Transformer论文中，采取的是将 Positional embedding 和 词向量信息进行相加。
**这种做法直觉上会导致信息的丢失**，但是Google的成果证明相加的效果也不错，这就是其中玄学的地方了。


<figure>
	<a href=""><img src="/assets/images/transformer/pe.jpg" alt="" width="600" heigh="400"></a>
    <figcaption><a href="" title=""> Positional embedding 和 词向量信息相加，得到模型的输入信息_</a></figcaption>::
</figure>

在Facebook的一篇论文 _Convolutional Sequence to Sequence Learning_ 中也用到了Positional Embedding的概念，但是与Google的这种通过公式计算的定义方式，Facebook 采取
的是 learned and fixed 的方法，这代表着训练一旦结束，这种方式只能表征有限长度的位置，无法对任意的位置进行建模。同时实验结果指出这两种形式的模型没有效果上的差别，
对比下来，那么明显是上面的公式定义的方式更为简单和轻便。

### 4.2 Weight Typing

在原论文中，有一段话提到Weight Typing的这个technique：
> Similarly to other sequence transduction models, we use learned **embeddings** to convert the **input**
tokens and **output** tokens to vectors of dimension dmodel. We also use the usual learned linear transformation
and softmax function to convert the decoder output to predicted next-token probabilities. **In
our model, we share the same weight matrix between the two embedding layers and the pre-softmax
linear transformation**

一开始我也对这段话不太理解，直到后来我看到知乎上的一片文章：[知乎专栏：碎碎念：Transformer的细枝末节](https://zhuanlan.zhihu.com/p/60821628)，里面对
Weight Typing 技术进行了详尽的描述，更加具体的建议参考这篇文章，这里我将作者的观点稍作提取：

假设在Decoder阶段输入和输出的词表大小为 $$C$$，隐层的输出大小为 $$H$$ (隐层的大小往往和embedding向量的大小一致)，那么，在输入阶段，则有一个 $$U \in R^{C \times H}$$ 的embedding矩阵，
而对于输出阶段（pre-softmax），则同样有一个 $$V \in R^{H \times C}$$ 的矩阵，用于将隐层向量转化为词表大小的向量，最后再用于softmax归一化；

而所谓的 Weight Typing 技术，则是将这两个矩阵直接共享参数。即 $$U = V$$。在反向更新的过程中，这些共享的参数会被两种不一样的梯度进行了两次的更新。但是值得注意的是，**embedding 和 pre-softmax
层依然是不同的两个层，因为它们的bias不一样。**，采取 Weight Typing 技术的好处，主要在于：

* $$U$$ 矩阵由于层叠加的问题，反向传播过程中，训练不如矩阵 $$V$$ 充分。将两者绑定在一起，可以训练得到效果更好的矩阵；
* Weight Typing 技术能够有效的减少模型的参数。（特别是在词表较大的情况下更为明显）
* 模型参数变少了，则模型收敛会更快。


除了在Decoder的embedding层和pre-softmax层做参数共享，有些情况下还会令 encoder 的 embedding 层和 decoder 的embedding 层做参数共享。例如在智能对话的场景下，
encoder 和 decoder 的词表是一致的。甚至在翻译问题上，尽管 encoder 和 decoder 的词表不一致，但是如 Transoformer 在英法和英德翻译的问题上，混用了源语言和目标语言的词表，
因此使用了升级的 weight-typing 技术：**TWWT（Three way weight tying）**，把encoder的embedding层权重，也加入共享：

{% highlight python %}
if model_opt.share_embeddings:
    tgt_emb.word_lut.weight = src_emb.word_lut.weight
{% endhighlight %}

### 4.3 Scaled Attention

主流的 Attention 主要分为 Additive 和 Dot-Product 两种形态：

* Additive: $$score(K, V) = W.dot(tanh(Q + V))$$
* Dot-Product: $$score(K, V) = K.dot(V)$$

上述两种方式的参数数量和运算复杂度相差不远，在 Transformer 论文中选择了 Dot-Product 的形式，(**在实际的运行效果上 dot-product 的形式更快，得益于深度优化的矩阵乘法运算**)
并且添加上 scaled 因子 $$\sqrt{d_k}$$。

$$
Attention(Q, K, V) = softmax(QK^T/\sqrt{d_k})V
$$

其主要考虑的原因是在论文[《Massive Exploration of Neural Machine Translation Architectures》](https://arxiv.org/pdf/1703.03906.pdf)中对
Dot-Product 和 Additive 两种Attention的形式进行对比，其中发现在 $$d_k$$ 较大的时候， Dot-Product 形式会效果更差，主要分析的原因在于
**$$d_k$$的增大，导致点击结果上限变得很大，将点击结果推向softmax函数的平滑区，影响了训练的稳定性。**，因此，Transformer 论文中，通过添加了缩放因此 $$d_k$$，
来对这个问题做进一步的调节。


### 4.4 Residual Network & LayerNorm
在 Encoder 和 Docoder 中每个Sublayer 的讲解中，我遗漏了一个重要的关键点： **每个sublayer，都会接上一个 residual-shortcut 和 Layer Normolization 层**。具体如下图：

<figure>
	<a href=""><img src="/assets/images/transformer/lm.jpg" alt="" width="500" heigh="500"></a>
    <figcaption><a href="" title=""> 每个sublayer中的residual-shortcut 和 layer-normalization 层</a></figcaption>
</figure>

对于上图，用公式来表示就是 $$output = LayerNorm(X+Z)$$。对于 residual-shortbut，可以参考本博客之前的一篇博文 [《Residual Network详解》](http://dreamingo.github.io/2018/02/residual_net/)，
总体而言，有以下的好处：

* residual-shortcut 的存在，在计算量上既没有增加多少，同时能够令梯度在反向更新的时候能够在不同层次之间自由的流通，使得构建和训练**更深**的网络时，模型更容易收敛。
* redisual中 element-wise 加的方式，残差的引入使得模型忽略了函数主体部分，从而更加专注于微小的变化。

<figure>
	<a href=""><img src="/assets/images/residual/optimize.png" alt="" width="600" heigh="400"></a>
</figure>

而对于 [Layer Normalization](https://arxiv.org/abs/1607.06450v1)，我在这里直接贴出两篇引用文章，系统框架性的介绍了深度学习中 Normalization 的原理和作用：

* [知乎专栏：Juliuszh - 详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)
* [知乎问答：怎样看待刚刚出炉的 Layer Normalisation 的？](https://www.zhihu.com/question/48820040)

珠玉在前，我这里就不便多说。我在这里主要讲述下：**为什么Transformer中使用Layer Normalization 而不是 Batch Normalization**。

在探讨Transformer之前，我先来说说 RNN 中为什么不适合使用BN。
其实也有一些论文提出在 RNN 网络中使用 BN，例如[Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025)，
但总体而言不常见，而且效果较差。主要原因在于：

* 天然的Batch Normalization是在CNN任务中提出的，需要较大的 batch-size 和 充分的shuffle 来保证统计的可靠性。并且记录全局的 $$\sigma$$ 和 $$\mu$$ 来供预测任务使用。
但是对于RNN而言，每个时序 timestep 的参数权重是共享的，所以每步统计下来的参数，是完全的不一致，每步之间是不具备通用性的。如果每个时序分来来统计，则会导致原来简单的BN流程非常的复杂。

* RNN 中实际上序列是变长的（尽管可能通过 pad 0 补充到同样的长度），这会导致长句子的统计不足。

对于 Transformer 而言，尽管其没有了时序上的困扰，而且更像是一种大力出奇迹式的特征抽取网络，为什么不能用BN呢？我觉得主要的问题还是出在上面提到的第二点中，输入的序列实际上是变长的。
同时，在一篇paper [Training Tips for the Transformer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf) 中也有提到尝试在 transformer 中使用 BN，但是实验结果同样的不理想(模型无法收敛)：

> Applying batch normalization on RNN is difficult. Transformer does not use RNN, but still we were
not successful in switching to batch normalization (and possibly ghost batch normalization) **due to NaN loss errors.**

同样的，在参考引用3 中，作者也尝试将 Transformer 中的 LN 层去掉，模型同样是无法有效的收敛，可见 LN 层对于 Transformer 模型的重要性。


## 0x06. 个人感想
未完待续...


## 0x04. 引用
* [1].[Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
* [2].[Jay Alammar - The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
* [3].[苏剑林 - 《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765)
* [4].[知乎专栏：小莲子 - 碎碎念：Transformer的细枝末节）](https://zhuanlan.zhihu.com/p/60821628)
* [5].[从 Transformer 说起](https://tobiaslee.top/2018/12/13/Start-from-Transformer/)
* [6].[知乎专栏：张俊林 - 放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
