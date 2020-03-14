---
layout: post
title: "聊聊神经网络翻译系统(NMT)的健壮性"
modified:
categories: 机器学习 
description: "deep learning"
tags: [nlp nmt deeplearning robustness]
comments: true
mathjax: true
share:
date: 2020-03-13T17:53:10+21:12
---

> 本文主要记录在实际工作和阅读相关论文中，有关提升NMT健壮性（robustness）的一些心得和建议；

## 0x01. 前言:

深度神经网络，特别是2017年Transformer横空出世后，NMT（Neural Machine Translation）领域得到了飞速的发展，相关的论文在这几年下来如雨后春笋般茁壮成长，不少语系在各项比赛（如WMT）的BLEU-Score也不断的被刷新；
在这些干净、in-domain的比赛测试集中，NMT往往能够得到令人满意的结果。但同时，不少发现也证明了NMT对输入噪音的敏感性高，往往很容易因为输入中有少量扰动(Perturbation)而导致输出结果和原来相比大相径庭。

<figure>
	<a href=""><img src="/assets/images/nmt_robustness/google_trans_errors.jpg" alt="" width="700" heigh="400"></a>
	<figcaption><a href="" title="">NMT的抗噪能力：强如Google翻译，在源语句中插入了两个表情符号，模型的输出也会有大幅的变化</a>.</figcaption>
</figure>

因此，在实际工业界的机器翻译模型中，比起拼死拼活的提高模型在干净语料上 0.x 的BLEU分数上，大家更关心的是模型在面临实际落地场景的时候的健壮性。这些实际的落地场景，往往会涉及大量的C端用户生产的语料（聊天，query，评论等），
充斥着各种的错别字、缩写、俚语，表情符号，语法错误、unkown字符等等的问题。除了在NMT系统前接上特定的处理任务（例如纠错、缩写术语替换等），NMT系统本身也得有着强悍的抗噪能力。

提高NMT模型的抗噪能力，无论是学术界paper和实际工作中，基本都分为以下的三个方面：

1. 数据层面：针对特殊的场景，利用数据增强的手段，生成大量对抗样本；
2. 模型层面：修改模型的学习目标，利用对抗(Adversarial Training, Contrastive Training) 的方式，结合对抗样本。提升模型鲁棒性；
3. Decode层面：更多是一些小技巧，例如 model-average，model-emsemble（多个NMT模型，或者NMT模型和LM模型打分Rerank），多用于比赛中。

下面我们来简单聊聊，在**数据层面**和**训练层面**我们能做的一些工作（更多论文可以参考THU的[MT-Reading-List#robustness](https://github.com/THUNLP-MT/MT-Reading-List#robustness)）

---

## 0x02. 对抗样本：

在开始之前先说结论：**生成对抗样本，是实际上最简单有效的办法。根据实际场景的特征，设计不同的对抗样本，能够玩出很多花样，解决非常多的实际问题。** 而对抗样本的生成，往往可以分为以下两个方面：

* **人工精心合成的语料：** 带有强目的性，往往是训练者先预设好一定的目标（例如：让模型学习出unk符号的复制、提高错别字的抗噪能力等），根据这些目的人工合成语料，灌输给模型训练，
让模型能够捕捉到合成数据中的特征，从而达到预先设计的目的。
* **模型合成：** 通过训练好模型生成，往往会在生成的时候加上一定的**噪音**（例如 Sampled Back-Translation）。相比于人工合成，这些语料会更加的自然和广泛。

### 2.1 人工合成语料：

人工的合成语料，往往会在已有的平行语料进行修改。在某些涉及到要同时修改源语句和目标语句的时候， 往往我们会先通过 `fast-align` 等工具先获取一份翻译语对的条件概率对齐词表。通过对齐词表，我们才能够同时定位到某个词语在源语句和目标语句的位置，然后根据位置再做出修改措施。

<figure>
	<a href=""><img src="/assets/images/nmt_robustness/align_table.jpg" alt="" width="550" heigh="550"></a>
	<figcaption><a href="" title="">样例：根据提供的对齐词表，就可以同时修改源句和目标句对应位置的单词（插入特殊符号）</a>.</figcaption>
</figure>

我在这里就列举几个**常见**NMT领域的人工合成抗噪语料方式，在实际上，往往还会有根据自身场景和任务特点进行更细致的优化：

#### 2.1.1 错别字对抗：

错别字，象音字这种是在社交网络、搜索 query 等场景中最常见的文本噪音。因此，在对抗数据生成层面中，往往也会根据这点进行优化。旨在令模型能够减少对这些错别字对最终NMT系统输出的影响。常见的做法往往会有：

* 通过对源语句随机位置的单词内部，对字母进行一定程度上的删除、插入和交换。例如保证在编辑距离为1~2的情况下进行修改；
* 在类似中文等语言中，对源语句中常见的单词，利用象音字进行一定程度上的替换。例如：苹果 -> 平果

<figure>
	<a href=""><img src="/assets/images/nmt_robustness/misspell.jpg" alt="" width="550" heigh="350"></a>
	<figcaption><a href="" title="">样例：根据编辑距离 or 象音的合成噪音语料</a>.</figcaption>
</figure>

#### 2.1.2 随机替换、交换、删除和插入等：

随机的对源句的单词进行替换、交换、删除和插入等行为，同样也是希望在源句中加入小量的扰动，提高模型对错误语法、词法的包容性。这种方式的扰动总体而言比较**粗暴**，过火了话很可能导致模型的精度下降。在控制上需要更加的小心和细致。

* 替换：随机选择源句中的部分单词，利用这个单词的近义词 or 词向量相近的单词进行替换；
* 交换：随机选择源句中的部分单词，将这些单词和它在范围窗口内的另外一个单词位置交换；
* 删除：概率性的丢掉源句中的一些单词（除了随机外，更细腻的控制例如丢掉介词，高频词等）

<figure>
	<a href=""><img src="/assets/images/nmt_robustness/demo.jpg" alt="" width="550" heigh="550"></a>
	<figcaption><a href="" title="">简易demo</a>.</figcaption>
</figure>

#### 2.1.3 Unk符号和特殊符号对齐：

在实际的语料中，输入的源句子往往会包含一些OOV的符号（如表情符号，来自别的语言体系的字符），又或者是特殊符号（例如颜文字等）。例如：

```python
# 包含别的语言体系的字符：
泰米尔语[注 1]（தமிழ் Tamiḻ）是一种有超过二千年历史的语言，属于达罗毗荼语系，通行于印度南部、斯里兰卡东北部。
# 包含OOV的表情字符：
"💕2019 SPECIAL HEART TOUCHING COLLECTION EVER❤️BEST OF THE YEAR 2019❤️|"
# 语句中包含各种奇怪的ascii字符，不常见于正常文本：例如Twitter文本：
"#Breakings!#  | ~~Here is a bad news @Moviecats ^_^ ~~"
```

因此，对于这些特殊的字符，我们都希望模型在正确翻译正常文本的同时，能够把这些符号**从源句copy到目标句中。**，本质上是希望强化模型 Copy 的机制（这个时候Shared input和target的embedding往往会有更好的结果）。

在正常偏干净的文本中训练出来的模型，面对上面的这些例子往往会输出比较糟糕的结果。一方面因为`<UNK>`符号的稀疏，导致所对应的向量训练不充分。因此我们可以发现模型在decode阶段出现`<UNK>`符号时，输出的整句话受污染扰动的可能性很大；另外一方面NMT的decoder本质上也是一个语言模型，因为与训练样本的schema不太一致，这些不规则的符号很可能会在decode阶段被忽略掉。

因此，我们可以通过在源平行语料中，在对齐的位置随机生成一些非常正符号(除了随机外，建议更精心的设计和模拟落地场景的噪音)，
辅助模型学习在面对这些非正常符号的时候，学习出 Copy 的机制（这种Tag-Copy的方式也可以应用于NMT系统的结果干预上哦）

```python
# 原平行语料
"Small perturbations in the input can severely distort intermediate representations and thus impact translation quality of neural machine translation (NMT) models."
"输入中的小扰动会严重扭曲中间表示，从而影响神经机器翻译（NMT）模型的翻译质量。"

# 合成平行语料1：
"Small [@- perturbations #%~ in the input can # severely # distort intermediate representations and thus impact translation quality of neural machine translation (NMT) models."
"输入中的小 [&- 扰动 #%~ 会 # 严重 # 扭曲中间表示，从而影响神经机器翻译（NMT）模型的翻译质量。"

# 合成平行语料2：
"😄😄Small perturbations in the input can severely distort intermediate representations and thus impact translation quality👏🏻👏🏻 of neural machine translation (NMT) models."
"输入中的😄😄小扰动会严重扭曲中间表示，从而影响神经机器翻译（NMT）模型的翻译质量👏🏻👏🏻。"
```

#### 2.1.4 标点补充：
在人为输入的场景中，经常会出现一整段话没有停顿没有标点，如果模型在训练过程中也学习出了严谨的标点对齐能力，并把原文直接翻译过来，难免也会导致输出结果话语过于生硬。

```python
# Google Translation 的例子：模型学习出更加自然流畅的断句方式。
"今天中午我在饭堂看到他了他长的真帅啊"
"I saw him in the dining room today at noon. He is so handsome."
```

如果发现训练得到的NMT模型在标点补充、停顿方面的能力表现不足的话，那么也可以通过从原来的平行语料中，通过随机删除源句的一些标点符号，强迫模型学习出补充标点的能力。


#### 2.1.5 特殊格式文本：

在实际应用场景中，特殊格式的文本种类繁多。而最常见的，莫过于是 网址、email地址、时间日期、数学公式等。对于这类型的特殊格式文本，最常见的就是希望模型能够直接将其 Copy 过来，不进行任何的翻译。例如：

```python
"helloworld@gmail.com    -> helloworld@gmail.com"
"https://helloword.com/neural/machine/translation.html   -> https://helloword.com/neural/machine/translation.html"
```

对于这类型的特殊格式文本，实际工作中往往会有非常多的处理方式，例如在翻译模型的前后进行特殊前后处理（标记、替换、对齐等）。但是提升模型在应对这类型特殊文本翻译能力往往也会非常重要。
因此，一方面我们可以新增生成大量的这些合成语料，另外一方面也可以通过在现有平行语料中插入合成噪音实现。

```python
# 新增噪音：
"helloworld@gmail.com    -> helloworld@gmail.com"
"https://helloword.com/neural/machine/translation.html   -> https://helloword.com/neural/machine/translation.html"

# 插入噪音

"Small perturbations in the input can severely distort intermediate representations helloworld@gamil.com and thus impact translation quality of neural machine translation (NMT) models."
"输入中的小扰动会严重扭曲中间表示 helloworld@gmail.com，从而影响神经机器翻译（NMT）模型的翻译质量。"
```


### 2.2 模型合成语料：

在大规模的模型层面合成噪音语料，最**著名实用**的莫过于2018的论文 [Understanding Back-Translation at Scale](Understanding Back-Translation at Scale)。通过在Back-Translation的decode阶段，
对logits的输出进行概率采样 或者 beam+noise 的方式引入『非最优』噪声，达到了生成『同样的结果，不同表达』多样化合成语料的目的，提升模型的鲁棒性。

<figure>
	<a href=""><img src="/assets/images/nmt_robustness/bt.jpg" alt="" width="650" heigh="450"></a>
	<figcaption><a href="" title="">Understanding Back-Translation at Scale</a>.</figcaption>
</figure>

除了上面的经典论文之外，也有不少是从模型层面（对抗损失函数，领域信息等），构造出来令模型更容易受扰动的对抗语料，例如下面的这篇论文，这里不做展开：

Yong Cheng, Lu Jiang, and Wolfgang Macherey. 2019. [Robust Neural Machine Translation with Doubly Adversarial Inputs. In Proceedings of ACL 2019.](https://arxiv.org/pdf/1906.02443)

- - -

## 0x03. 模型层面：

同样的在开始之前，先总结一番我认为的观点：**目前模型层面提高模型鲁棒性的顶会论文也不少，都通过各自的手段来论证有效性。但实际工作体验下，并不会比对抗样本的方式高明多少，总体感觉ROI偏低。可以挑选一些有代表性的工作，结合对抗样本生成来尝试。**

在这里， 我介绍下我认为比较代表性或者有实战意义的一些论文，仅为个人观点：

### 3.1. Adversarial Training

在这里我利用18年腾讯的一篇ACL论文来讲讲：[Towards Robust Neural Machine Translation](https://www.aclweb.org/anthology/P18-1163.pdf)，如何在机器翻译的过程中引入对抗学习的思路提升模型鲁棒性。

论文的宗旨，是希望模型在面对输入扰动(perturbations)的时候尽可能的不受影响（perturbations consistent）。可以分为以下两点来开展：

* Encoder：对于正常输入 $$x$$ 和 扰动输入 $$x'$$，encoder的输出向量集合 $$H$$ 和 $$H'$$ 尽可能的相似。
* Decoder: 对于不同的Encoder输出$$H$$ 和 $$H'$$，decoder均能够成功预测出正常输出 $$y$$

<figure>
	<a href=""><img src="/assets/images/nmt_robustness/tc.jpg" alt="" width="400" heigh="400"></a>
	<figcaption><a href="" title="">在NMT模型中嵌入对抗学习的思路</a>.</figcaption>
</figure>

看到上图，核心在于多添加了一个判断器(Discriminator)，最终的学习目的，是希望Discriminator无法分辨出生成器（Generator，在上图中其实就是Encoder），生成出来的编码向量$$H$$ 和 $$H'$$的区别。因此最终模型的训练目标，本质上变成了同时训练学习对抗目标和NMT目标的一个多任务学习目标。


<figure>
	<a href=""><img src="/assets/images/nmt_robustness/obj.jpg" alt="" width="300" heigh="300"></a>
	<figcaption><a href="" title="">最终的学习目标：同时最小化对抗损失和NMT的损失函数</a>.</figcaption>
</figure>

### 3.2 Contrastive Learning:

这里主要介绍19年ACL的一篇论文：[Reducing Word Omission Errors in Neural Machine Translation: A Contrastive Learning Approach](https://www.aclweb.org/anthology/P19-1291.pdf)。这篇论文主要介绍通过Contrastive Learning（差异化学习）的手段来减少NMT模型漏译的情况。

过往解决NMT模型漏译的情况，一般是通过 Attention-Weight-Coverage 机制来开展的。而这篇论文另辟蹊径，思路比较新颖和朴素：

1. 对于正常的平行语料$$(x, y)$$，通过一些手段生成漏译的结果$$(x, y')$$:
    * 随机丢掉$$y$$中的一些单词；
    * 概率性的重点丢掉如介词、高频词等这些单词。

2. 对于正常的平行语料 $$(x, y)$$，希望模型能够学习出更高的置信概率P；而对于漏译的人工语料 $$(x, y')$$，则希望模型应该学习出更低的概率P'。
更准确而言，应该是尽可能的使得 $$P - P'$$ 的数值尽可能的大，这就是差异化学习的核心。
3. 通过这种人工合成噪音，同时接入差异化学习框架的方式，令模型减少漏译的情况。

这种思路还有一个更值得看好的点，论文指出往往只需要训练好NMT模型后，再接入Contrastive Learning的框架，finetune几百步即可。算是一种非常经济实惠，高效的手段了。


### 3.3 其他：

除了上述比较重型的一些改动，还有一些针对特定问题的有趣模型改动。例如百度19ACL的这篇论文：[Robust Neural Machine Translation with Joint Textual and Phonetic Embedding](https://www.aclweb.org/anthology/P19-1291.pdf)。同样是解决输入中象音字的噪音问题。模型通过在输入embedding层，除了输入单词向量外，还增加了语音学上的embedding（例如这个单词的拼音信息）。通过这个手段提高模型针对象音字时候的抗噪能力。

---

## 0x04. 总结：

机器翻译的健壮性，这是一个在实际工业界落地时候非常值得重视的方面。上文从数据生成、模型等层面阐述了不同的手段。个人觉得总体而言：

1. 大规模的模型生成噪音样本（例如Sampled Bcak-Translation）可以作为整个NMT系统抗噪性的基石；
2. 人工精心合成的对抗样本，则可以作为针对特定问题or落地场景，定向爆破困难的利器；
3. 而对于模型层面的对抗改动，则需要精心挑选ROI高，有实际落地意义的工作，达到锦上添花的作用。


## 0x05. 引用：

1. Yong Cheng, Zhaopeng Tu, Fandong Meng, Junjie Zhai, and Yang Liu. 2018.[Towards Robust Neural Machine Translation. InProceedings of ACL 2018.](http://aclweb.org/anthology/P18-1163)
2. Zonghan Yang, Yong Cheng, Yang Liu, Maosong Sun. 2019. [Reducing Word Omission Errors in Neural Machine Translation: A Contrastive Learning Approach. In Proceedings of ACL 2019.](https://www.aclweb.org/anthology/P19-1623)
3. Vaibhav Vaibhav, Sumeet Singh, Craig Stewart, and Graham Neubig. 2019. [Improving Robustness of Machine Translation with Synthetic Noise. In Proceedings of NAACL 2019.](https://arxiv.org/pdf/1902.09508.pdf)
4. Hairong Liu, Mingbo Ma, Liang Huang, Hao Xiong, and Zhongjun He. 2019. [Robust Neural Machine Translation with Joint Textual and Phonetic Embedding. In Proceedings of ACL 2019.](https://www.aclweb.org/anthology/P19-1291)
