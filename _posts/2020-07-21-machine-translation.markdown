---
layout: post
title: "跨越语言的障碍：让机器翻译助力内容生产"
modified:
categories: 机器学习 
description: "deep learning"
tags: [nlp nmt deeplearning machine-translation]
comments: true
mathjax: true
share:
date: 2020-07-20T17:53:10+21:12
---


## 0x01. 背景：
部门产品出海深耕多年，主战场是两印（印度和印尼），同时支撑东南亚、俄罗斯、中东等多个空军国家。除了提供自身便捷快速的工具属性外（搜索、资源访问、资源下载、导航、网盘等），还致力于给用户提供便利的信息内容获取体验（信息流、推送、垂直站点等），丰富的内容涵盖文章、短视频、小视频、GIF，比赛直播等多种形态，满足用户对内容不同纬度需求的满足。

在最主要的主战场 - 印度，有12种语言的使用人数超过一千万，并且不同的语言地域分布广泛，语言间文化差异巨大。同时很多小语种的优质内容稀缺，在有限的成本和人力的情况下往往难以生产到足够的内容满足对应的用户群体。下面右图是今年年初信息流不同语种的视频内容每日入库量的对比图。稀缺语种的入库量，大概只有主流语种的五分之一到十分之一不等。因此，利用机器算法来辅助内容生产便有了合适的契机。


<figure>
	<a href=""><img src="/assets/images/machine_translation/lang_sparse.jpg" alt="" width="700" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

在去年6月份初步调研一番之后，发现目前公司内部翻译和业界的Google-Translate因为覆盖率和翻译效果问题，并不能满足目前我们的业务需求，经过权衡后，我们决定立项机器翻译项目，不做大而全的通用翻译，立足于解决业务特定领域的翻译问题，快速的验证机器翻译对内容生产的辅助作用。
* **项目目标：** 通过机器翻译的手段，辅助内容生产，缓解小语种内容供给的问题；
* **翻译目标：**
    * **翻译语向：** 印度主流语种 -> 小语种；
    * **目标1：** 印度的热门主流领域（政治娱乐体育社会等）翻译效果超过 Google-Translate；
    * **目标2：** 特定专业领域（例如板球）等翻译效果大幅超过 Google-Translate；


## 0x02. 业务成果：
项目从19年7月成立后，前后共三名同学参与。截止目前为止，共完成了 英语-印地语、英语-泰米尔语 以及 英语-泰卢固语三个语向模型的研发和上线；

### 2.1 离线评测：
> 翻译结果的好坏，按照严复老先生的说法，应该遵循 『信、雅、达』这三个方面。所以在实际的项目过程中，我们离线采取 BLEU 指标来衡量翻译结果的准确率和流畅程度（信&达），采取人工评估来衡量翻译结果的体感效果（雅）

#### 2.1.1 BLEU
**BLEU**作为机器翻译业界和比赛中最常用的评估指标，主要通过翻译结果和译文之间连续 ngram（一般一元到四元） 的重合程度来衡量翻译结果的准确率和流畅程度。为了减少因为翻译多样性所带来文本重合指标上的偏差，在重要的测试集上我们采取了一个源句，三个reference译文的方式来进行评估。

可以看到，两个语向上的通用测试集上，我们的分数均比业界最好的 Google-Translate 有2-3分明显的提升；而在专业的领域（板球球评解说）上，提升更是异常显著（BLEU 10+）；

<figure>
	<a href=""><img src="/assets/images/machine_translation/trans_perf.jpg" alt="" width="700" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

#### 2.1.2. 人工体感评测：
在上线前，为了更进一步从人类的体感上来评估翻译结果的好坏，我们制定了详细的打分标准，将翻译结果的好坏划分为 Perfect，Good，So-so 和 Not-OK 这四档，并且同时邀请三家印度的专业供应商来进行打分评价，以此来减少不同人评估上的偏差。

可以看到，在专业的板球解说球评的翻译中，Google-Translate的优良率只有27.2%，这是远远达不到上线标准的。而我们项目大幅提升到 88.7%的同时，也获得了专业小编的认可。

<figure>
	<a href=""><img src="/assets/images/machine_translation/human_trans_perf.jpg" alt="" width="700" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>


### 2.2 线上业务结果：
#### 2.2.1 板球站线上文字直播翻译：

板球是印度的国球，在印度有着非常深厚文化和狂热的国民热爱；部门产品立足于从赛事新闻、直播、比赛周边、社区讨论等纬度来服务印度群众，比赛期间峰值DAU甚至一度超过1300w+；

受制于有限的资源，目前板球站在比赛期间仅有一名专业的英语解说员进行实时线上文字直播。为了服务更多其他语种的用户，同时考虑到板球解说总体偏短，文字专业性强，具有体育赛事直播中固定的解说套路。因此利用机器翻译来生成其他语种在技术上是可行的。


经过三个月的奋战，我们从0到1完成了en-hi的板球球评翻译模型的上线，并经过详细的ab实验对比，为板球站的文字直播功能带来了明显的业务收益：
* 文字直播渗透率提升 6.0%（ 49.8% vs 47% ）
* Hindi用户人均使用时长提升 2%（0.12分钟）
* 板球站次留提升相对值 14.0% （15.17% vs 13.31%）


<figure>
	<a href=""><img src="/assets/images/machine_translation/cricket_demo.jpg" alt="" width="300" heigh="500"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

#### 2.2.2 信息流视频字幕生成翻译 + 标题翻译

在信息流中，主流的语言，每日有着充足的视频生产供应（1w+）。但是对于小语种而言，无论是从爬取还是社会化生产的角度而言都是非常稀缺的。那么一个比较自然的想法，就是借助机器翻译的能力，将主流语种的视频搬运到小语种频道中，丰富不同语种用户对于视频消费的满足；

在项目中，目前我们主要立足于将英语视频搬运到小语种中。 短视频标题一般简明扼要，机器翻译后的体感尚佳。而对于某些品类的视频，还会利用语音识别的技术生成英语字幕，然后再翻译成对应的小语种后来生成双语字幕，辅助用户阅读理解。


<figure>
	<a href=""><img src="/assets/images/machine_translation/subtitle_demo.jpg" alt="" width="600" heigh="500"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

在信息流实验中，我们通过每日将大量的热门消费英语短视频翻译后，重新入库到泰米尔语频道中，因此来扩大视频生产规模。通过线上详细的ab实验对比，能够验证出机器翻译生产带来非常有效的业务价值：
大量机器翻译视频的加入，全面提升了整个Tamil语信息流用户的消费活力和深度（**人均消费PV +4.98%，人均消费时长 +5.49%，横屏视频消费CTR +5.19%），提升用户留存（新次留+1.79%，全次留+0.55%**）

#### 2.2.3 热点发现内容整合：
机器翻译还辅助对于热点发现的能力，通过将不同语言的热门事件关键词翻译后归一化，能够有效的对不同语言的同一个热门事件进行合并、整合；

---

## 0x03. 整体架构：
考虑到项目人力和目标，翻译项目的整体架构主要以围绕业务开展，最小功能可用为宗旨来搭建，共包含以下的几大模块：

<figure>
	<a href=""><img src="/assets/images/machine_translation/project_struct.jpg" alt="" width="800" heigh="700"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

### 3.1 数据挖掘&爬取：

> 机器翻译作为一个庞大的深度学习任务，充足的语料是整个项目成功的基石。数据的挖掘&爬取决定了最终效果的上限；总体而言，主要分为以下几个模块：

#### 3.1.1 平行语料：
平行语料是机器翻译模型的基础训练语料，包括输入源句以及翻译结果目标句。准确的平行语料是非常稀缺的资源，而我们要做的印度语种情况更是严峻。平行主要挖掘来源包括以下方面：

* 开源电影字幕：
    * 电影字幕是天然优质的平行语料，Opensubtitle是全球最大的开源字幕网站，通过爬虫爬取以及对应的清洗解析，我们获取到一定量级的对齐平行语料；
* Wikipedia 标题对：
    * 同一个item实体在wikipedia上往往会有不同语种的文章百科。文章内容往往难以对齐，但是标题却是准确无误的，而且wikipedia上挖掘得到的标题对，往往是专业词汇的标准翻译；
* 专业领域数据挖掘：
    * 针对特定的业务的术语翻译挖掘，例如球员、球队、电影演员翻译对照表等；
* 伪平行语料：
    * 我们构建了一个分布式的Google-Translate的爬取引擎，爬取吞吐量能够高达600w/天。这些爬取的伪平行语料在后续会通过 back-translation 的数据增强技术来训练模型；
* 其他：
    * 如 Tedtalks字幕、电子书译本、词典、开源数据集的收取等；
* 人工标注：
    * 对于专业领域（如板球球评直播），我们会采取少量的专业小编人工翻译标准的方式来辅助模型进行领域迁移；

除了通过爬取&挖掘的方式外，我们还尝试过参考 Facebook LASER 的思路，尝试将信息流内容库中数千万不同语言的文章，利用弱翻译的方式将不同语言的句子对齐到同一个向量空间中，从而进行自动化的平行语料挖掘工作。但总体下来后发现挖掘效率和对齐精度并不理想，投入产出比也不高。

#### 3.1.2 单语语料：

除了双语平行语料之外，我们还针对待翻译语种（如英语、印地语、泰米尔语等）挖掘来大量的单语文章语料。主要涵盖Wikipedia文章、信息流库存新闻文章、视频标题等。并利用这些单语语料进行如语言模型、back-translation、实体挖掘等相关任务；

### 3.2 离线数据处理体系：

> 如果说数据挖掘语料的数量是效果的基石，那么数据处理的质量则是最终效果的决定性因素。这一点在多语言的情况下更显得具有挑战性；

<figure>
	<a href=""><img src="/assets/images/machine_translation/post_process.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

离线的数据处理体系对接的上游任务丰富，例如模型的训练、在线服务以及离线的数据挖掘等任务。实际项目过程中，通过合理设计屏蔽了不同任务之间细微的差别，组件化不同逻辑的处理脚本，根据不同任务需求灵活自由的组装。离线的数据处理体系主要包括三个方面：
* 数据预处理&清洗：
    * 常见数据清洗处理，如不可见字符、html转义，标点处理、字符normlized等。
    * 语言相关类：如语言识别、分词断句以及特定语言体系的字符、词汇归一化、标点辅助对齐等问题；

* 数据质量检测：数据质量（Quality Estimation）体系进行了平行语料的翻译质量的自动评估和单语语料的语句通顺性评估。在实际项目中，我们结合自身数据的特殊性，同时也大量参考了前沿论文的做法，总体归结下来分为以下几个方向：

    * 基于规则：规则方法简单准确且高效，常见的例如有效语言字符比例、平行语料长度比例、是否过译等；
    * 基于统计模型：利用fastalign等对齐工具离线从平行语料中训练对齐模型，得出正反向的词汇概率对齐表格（如下图）。通过该表格可以计算出平行语句的对齐得分，并针对如特殊标点，数字、网址等特殊文本的权重；
    * 基于深度模型：例如通过语言模型的preplexity分数来判断单语语料的流畅长度，以及利用翻译模型的prediction-score来衡量平行语料的对齐程度；

<figure>
	<a href=""><img src="/assets/images/machine_translation/align_table.jpg" alt="" width="600" heigh="300"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>


* 数据增强：
    * 包括模板化的数据合成、数据加噪等数据增强的方式，在后面会详细讲述；

### 3.3 算法模型：

核心翻译模型主要以编码器解码器各6层的Transformer-base为主，模型训练主要以单机8xV100卡，配合混合精度技术进行加速。在算法层面，还会围绕着Transformer为主，根据业务特性进行的一系列优化，例如BPE-Dropout、相对位置编码的引入、改良后的schedule-sampling算法等。在后面章节会在详细介绍；

### 3.4 对外Serving服务：
由于我们的任务大部分都是离线的后台任务，对时延和吞吐量要求不高。这一块我们采取 TFServing on CPU的方式来做模型服务，前端搭建 WebServer 来进行基本的NLP处理等相关前置和后处理任务；

---

## 0x04. 核心技术挑战：

### 4.1. 小语种平行语料的数据稀疏
准确专业的平行语料在互联网中是比较稀缺的语料，而对于我们要做的印度语种情况更是严峻。总体而言我们所做的语种内容生态都相对封闭，能够有效利用的双语资源非常稀疏。在下图中列举了两个数据，其中第一个来自于全球最大的开源字幕站点 Opensubtitles，印度语种（hindi、tamil等）的有效双语字幕数量在所有语种中排名是非常靠后的；而另外一个数据则是Wikipedia上印度语种的卡片数量，这一定程度的反映出这些语种的内容生态的凋敝和封闭程度； 

<figure>
	<a href=""><img src="/assets/images/machine_translation/lang_sparse2.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

要解决这个难题，我们会从数据挖掘、数据增强以及算法模型这三个纬度上去入手。其中，数据挖掘的工作在上一个章节已经有所覆盖，在这里直接介绍后面两个主题。

#### 4.1.1 数据增强：
Back-Translation（回译）是机器翻译业界中使用最广泛的数据增强方式。其中下图对其工作原理有一个大概流程介绍：假设要训练一个英译中的模型，那么会先通过训练或者利用外部资源得到一个反向的翻译模型（中译英），通过在反向模型中输入真实的单语语料，利用翻译的手段生成一系列的合成语料。最终再利用这些合成语料来训练我们的目标模型（英译中）。

<figure>
	<a href=""><img src="/assets/images/machine_translation/back_trans.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

Back-Translation之所以有效，本质上是利用的翻译模型中encoder和decoder地位不对等的特性。就好像我们阅读的时候，粗略着看、跳着看也不会影响对内容的理解；但是在写文章的时候，则是逐个字写，文章的语义、语序等都需要细致的考虑。所以encoder端的输入可以是带噪音的合成语料，而decoder端的输入则必须是真实的单语语料；

在实际的项目过程中，我们参考了Google 2018年的论文[Understanding Back-Translation at Scale](https://arxiv.org/abs/1808.09381)，进一步的放大了这个『地位不对等』的特性。在反向模型解码的每一个decode-step，都并不一定选择对当前概率最大的输出，而是会从topk的logits中进行加权采样。这种类似加噪的方式一定程度上提升的合成语料的多样性以及提升模型的鲁棒性；

项目中，Back-Translation的来源单语语料主要是Domain相关的信息流文章和视频标题数据，涉及量级超过5000w+。与只有200w+的真实单语语料训练得到的模型相比，混入back-translation增强语料后的模型BLEU提升超过8.0，是我们整个项目效果的基石；


#### 4.1.2 预训练语言模型：

预训练语言模型是这两年NLP领域中非常火热的技术，例如BERT，通过自监督学习方式从大量的单语语料中先学习出有效的语言先验知识。但是 BERT 主体结构是一个双向的 encoder，对于encoder-decoder结构的翻译模型而言是不合适的。

所以在项目中，我们复现了2019年微软提出的[MASS](https://arxiv.org/pdf/1905.02450.pdf) 架构，本质上是通过改造 BERT 来联合训练 encoder和decoder，适应生成类的NLP任务。具体的模型架构如下图：
* 模型通过共享词表、利用多种语言的单语语料来进行多语言联合训练，其中通过在输出层添加 language embedding 层来协助模型区分不同的语种；
* 与BERT类似，自监督设计的方式是通过预测被掩盖的连续token，学习单语语料中的通用语言知识：
    * Decoder端仅会输入被掩盖的token，这样有助于在decoder预测时候更多的依赖于encoder的内容抽取理解，提升联合训练的效果；
    * Token设计为连续掩盖，本质上是为了 一定程度上增加Decoder在预测的时候对上文的理解能力；

<figure>
	<a href=""><img src="/assets/images/machine_translation/mass.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

在实际的项目中，引入预训练语言模型，与基线组相比，对翻译结果提升非常明显；如果训练语料中也引入5000w+的 Back-Translation语料，虽提升效果会被明显削弱，但还是能起到锦上添花的作用，主要原因是两种优化策略之间存在一定的重叠性：
* 【预训练 + 真实语料】  VS 【基线组（200w真实双语语料）】，BLEU +6.3
* 【预训练 + 真实语料 + 增强语料】 VS 【真实语料 + 增强语料】，BLEU + 0.4


### 4.2. Domain-Adaption：通用领域到专业领域的效果优化
> 特定领域的翻译效果是我们项目的立足根本，在这里以板球解说翻译为例，阐述训练好的通用领域模型如何迁移到专业领域中。

#### 4.2.1 数据层面：
有效的平行语料稀少，而专业领域的则更加的稀疏。但高效足量的数据则是领域模型效果的根本：
* 少量专业人工语料标注（4w+）：保障语料分布的充分性和覆盖率；
* 专业领域知识库建立（球员、球队、专业术语翻译对照表等）
* 模板化数据合成：我们发现板球球评解说中，文字之间往往是具有一定的格式套路（如下图），这种套路就给予了通过模板来做数据合成的可能性。令我们的标准语料从4w+扩充到80w+；

<figure>
	<a href=""><img src="/assets/images/machine_translation/cricket_temp.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>


#### 4.2.2 算法训练：
在算法模型层面，项目中采取 Mixed-finetune 的方式进行迁移学习，其中：
* 专业语料采样混合通用语料，有效减少 overfitting 的问题；
* 语料选择：通过分类任务or向量相似的方式，来挑选跟目标板球领域相关的通用语料（如体育类），提升迁移的效果和幅度；

此外，参考BERT在句首加入分类标签，我们也在翻译模型的句首，主动加入领域标签（Domain-label），通过全局Attention的方式影响生成，协助模型更有效的分别领域语料和通用语料的不同；

<figure>
	<a href=""><img src="/assets/images/machine_translation/pre_label.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

### 4.3. 模型鲁棒性
鲁棒性是工业界的翻译模型中非常重要的一环，因为在实际业务中所面临的语料（如下图这个待翻译的视频标题），往往会充斥着各种各样的脏数据，例如表情符号、缩写、俚语、错别字、多语言混写等噪音。

<figure>
	<a href=""><img src="/assets/images/machine_translation/noisy.jpg" alt="" width="600" heigh="200"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

基于神经网络的翻译系统对这些噪音是非常敏感的，往往会因为语句中含有一些特殊的噪音而导致翻译质量产生大幅的下降。下图是Google-Translation的例子：输入端是两段一模一样的话，而其中一段话间插入了两个表情符号，可以看到模型的输出产生了剧烈的变化，并且两个表情符号也没有被有效的翻译出来：

<figure>
	<a href=""><img src="/assets/images/machine_translation/gg_trans.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

提高NMT（Neural Machine Translation）模型的抗噪能力，无论是学术界paper和实际工作中，基本都分为以下几个方面，在我们项目中，基本也从这三个纬度入手：
* 数据层面：针对特殊的场景，利用数据增强的手段，生成精心设计的对抗样本；
* 模型层面：修改模型的学习目标，利用对抗(Adversarial Training, Contrastive Training) 的方式，结合对抗样本。提升模型鲁棒性；
* Decode层面：更多是一些小技巧，例如 Model-Average，Model-Emsemble（多个NMT模型，或者NMT模型和LM模型打分Rerank），多用于比赛中。实际工业届中则更多引入后处理环节；

#### 4.3.1 数据层面：数据增强
在实际项目过程中，数据增强的这种方式ROI高，是最简单有效的方式，具体体现在：
1. 大规模的模型生成噪音样本（例如Sampled Bcak-Translation）可以作为整个NMT系统抗噪性的基石；
2. 人工精心合成的对抗样本，则可以作为针对特定问题or落地场景，定向爆破困难的利器；例如：
    * 人工规则的错别字合成（基于编辑距离or错别字表）
    * 平行语料输入端单词的同义词（WordNet、词向量等）随机替换、删除、交换等操作；
    * 根据对齐词表，在平行语料对齐位置同时插入噪音，模拟视频标题 or Twitter噪音；

<figure>
	<a href=""><img src="/assets/images/machine_translation/anti_noisy.jpg" alt="" width="600" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

#### 4.3.2 线上服务：引入后处理流程
线上服务的后处理流程，更多时候则是类似于一个守门员的存在。对于特定badcase的报警通知、修复和规则化处理。具体体现在：
* 针对特殊文本（比分、Email等）与输入源句对齐；
* UNK 符号（表情、特殊字符、其他语种等）替换和翻译结果的干预替换；
* 翻译质量检测（例如严重漏译or过译），并降级处理；

#### 4.3.3 算法模型
* Subword正则化：
    * 在模型输入端为同一单词分解成subword的提供多变性，提升模型对处理罕见词、合体词、错别字等的容错能力；
    * 在我们项目中选择了[BPE-Dropout](https://arxiv.org/abs/1910.13267)算法，相比于[SentencePiece - ULM](https://arxiv.org/abs/1804.10959)的正则化方式，效率更快，并且对我们已有框架的侵入度更少。
    * BLEU效果稳定提升0.4+

<figure>
	<a href=""><img src="/assets/images/machine_translation/bpe_dropout.jpg" alt="" width="300" heigh="300"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>

* 差异化学习（Contrastive Learning）：
    * 在我们项目中，参考[论文](https://www.aclweb.org/anthology/P19-1623/)中提出利用差异化学习，在原有 Align-based 方式的基础上，来进一步辅助解决机器翻译中漏译的问题。我们通过一系列的方式构造出正负样本，同时修改模型的损失函数为 max-margin loss，本质上是令模型最大化正负样本之间的差别，这种方式的ROI较高，只需要在模型正常训练结束后finetune几百个steps即可；

<figure>
	<a href=""><img src="/assets/images/machine_translation/constractive_loss.jpg" alt="" width="700" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>
   
## 4.4. 模型可干预性

作为工业界的翻译模型，模型的可干预性则是日常运营过程中的重中之重。可以想象下当运营同学发现线上翻译引擎对某电影名字翻译有误时，想要干预输入片段中该专有名词的翻译结果。若此时我们没有合适的干预机制，则只能离线通过finetune的方式重新训练模型，这样的迭代效率在实际的项目中往往是不被接受的。

这一块我们考察了大量业界的做法，最终选择参考阿里达摩院的 [Code-Switching for Enhancing NMT with Pre-Specified Translation](https://www.aclweb.org/anthology/N19-1044/)这篇论文介绍的方法。本质上就是在训练阶段，让模型学会针对特殊符号or别的语种符号进行拷贝操作。具体的算法思路，别人珠玉在前，在这里就不展开介绍了。

下图简要的介绍了两种算法在实际线上的干预过程：

<figure>
	<a href=""><img src="/assets/images/machine_translation/code_switch.jpg" alt="" width="700" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>
   
---

## 0x05. 思考与展望：

这是今年年初提出对整个机器翻译项目未来的思考，主要还是围绕着加大业务落地效果和提高项目影响力两方面开展。其中，这个项目去年运转了大半年的时间，更多则是在中台化部门对小型不确定性需求难以快速覆盖的情况下，业务方一次快速的技术尝试。后续考虑到要进一步扩大的话，那一定还是会拥抱集团已有的技术架构和算法能力，站在巨人的肩膀上快速奔跑。

然而令人唏嘘的是，还未来得及进一步的思考和落地，印度业务却因为一系列的黑天鹅事件决定关停。那么此项目也因此告一段落。落笔之际，回顾过去，不禁感到惋惜。


<figure>
	<a href=""><img src="/assets/images/machine_translation/future.jpg" alt="" width="700" heigh="400"></a>
	<figcaption><a href="" title=""></a>.</figcaption>
</figure>
