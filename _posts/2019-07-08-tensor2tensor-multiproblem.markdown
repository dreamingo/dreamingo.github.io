---
layout: post
title: "Tensor2Tensor: 多任务学习(MultiProblem)源码阅读"
modified:
categories: 机器学习 
description: "deep-learning transformer source-code"
tags: [tensor2tensor transformer multi-problem]
comments: true
mathjax: true
share:
date: 2019-07-08T20:46:40+21:12
---

> 本文主要对 Tensor2Tensor 深度学习框架中，跟多任务学习（MultiProblem）相关的源码阅读。需要阅读者对 Tensor2Tensor 框架具有一定程度的了解

## 0x01. 前言：
随着2018年BERT开启了NLP领域中大规模预训练的大门，越来越多的NLP任务分解为 『two-stage』 两个步骤（预训练 + 下游任务）。除了预训练模式之外，多任务学习（Multi-Task Learning）也是今年NLP领域中火爆的话题。
对于一些数据量稀少的任务（例如小语种的机器翻译问题），利用额外任务中充足的语料联合训练是一个很好的办法。

例如：语言A(英语)和语言B(泰米尔语)的平行语料稀少，但是语言A(英语)和语言C(中文)平行语料很多，则可以采用共享NMT参数的方式同时学习多个翻译任务。其中，两个任务之间共享同一个encoder，而不同的任务之间有单独的 decoder 结构。
通过如此利用第三方的充足语料进行联合训练，能够令encoder对源语言的数据刻画能力有效的增强，从而一定程度上缓解稀缺数据的问题。

在上面的例子中，算是属于一个 one-to-many（共享一个encoder，多个decoder的模式）的架构。除此之外，还有 one-to-one（共享同一个encoder和decoder）和 many-to-one, many-to-many 等模式；

<figure>
	<a href=""><img src="/assets/images/multi_problem/setting.jpg" alt="" width="700" heigh="400"></a>
    <figcaption><a href="" title="">多种多任务训练的示意图，来自引用[3] </a></figcaption>
</figure>

## 0x02. T2T 中的多任务学习

**遗憾的是，在 Tensor2Tensor 框架中，MultiProblem 的定义，仅支持 one-to-one 模式。** 以训练模型为 Transformer 为例，one-to-one 模式，就是不同的任务之间共享同一个 encoder 和 decoder 任务。例如：

* 任务一：利用 wiki 语料训练的英语、泰米尔语的语言模型任务；
* 任务二：利用 英语-泰米尔语平行语料训练的机器翻译任务；

可以看到在上面的例子中，多个任务之间要求是相容，同时 encoder 和 decoeer 之间共享同一份大词表（英语 + 泰米尔语）。在这种模式下，利用拥有大量可以廉价获取的单语语料来训练语言模型（Language Model），以此来预训练模型，解决翻译问题中语料稀缺的问题。

初次看到这种 one-to-one 的多任务学习方法，心中是觉得这种模式比起 many-to-one 的方式更加的粗暴直接，心想不同的任务之间，连最后的 Softmax 层都共享是否合理？这其中有一种『大力出奇迹』的感觉。当然如果想要拥有类似上面 one-to-many 等架构，那就可能需要自己hack相关的源码，并自定义自己的模型了；这里就不再多说；

我们再次回到上面的这个例子中，语言模型和机器翻译问题本质上都是属于 `Text2Text` 的问题，两个任务之间的数据管理上是相通的。因此，既然 one-to-one 模式下，模型的架构没有变化，那么主要关心的点就在于 **在训练过程中，多个任务的数据如何调度和分配。**


## 0x03. T2T 中 Multiproblem 源码分析

在 Tensor2Tensor 中，分别有 `multi_problem.py` 和 `multi_problem_v2.py` 两个版本的多任务模块，两者的最主要区别在于针对数据的调度配置不太一样。接下来我们将详细的看看两者的区别。

### 3.1 MultiProblem

> 代码路径：`tensor2tensor/data_generators/multi_problem.py`

在开始之前，针对多任务学习，我先谈一些约定俗成的事情：

* 多任务中，定义的第一个任务，一般称为 primary-task；
* 一般情况下，primary-task 是一个语言模型(LM)类任务；
* 目前框架中多任务学习主要应用在NLP领域的任务，不同的NLP领域的任务，一般共享 primary-task 的词表；

#### 3.1.1 多任务数据的混合方式：

在Tensor2Tensor中，最终模型是利用 `estimator` 接口进行封装的，而数据则是通过预定义好 `input_fn` 输入到模型的。其中，函数`input_fn`的函数主体由自定义 Problem 的 `dataset` 函数作为入口。
那么，对于 `MultiProblem.dataset` 数据入口函数中，主要做了以下的几个事情：

```python
class MixingSchedule(object):
  """Available schedules for mixing datasets."""
  EXPONENTIAL = "exponential"
  CONSTANT = "constant"
  PRETRAIN = "pretrain"
```

1. 针对每个子任务 task ，调用其 `task.dataset()` 函数，得到一个列表，列表中的元素是每个任务的 `tf.Dataset.iterator`；
2. 针对每个子任务 task 的数据，为了能够使不同任务之间的数据统一进行处理，需要进行 normalized;
3. 根据超参中定义的数据融合方式 和 概率阈值，生成对每个子任务的采样概率策略；

在`dataset`函数中，每一个step进行一次的采样，本质上就是从上述1中的迭代器列表中选择一个，并调用其 `next` 函数。其中，在 MultiProblem 里面一共提供了三种数据采样的混合方式(mix schedule) ，分别是：

* **constant:** 每个任务拥有固定的采样概率。主要依赖于 `hparam` 中的超参：`multiproblem_schedule_threshold`，其中： 

    * Primary-Task 在每次采样的时候，有 `1 - multiproblem_schedule_threshold` 的概率被采样到；
    * 剩下的任务，在采样的时候，均分 `multiproblem_schedule_threshold` 的概率；

* **exponential：** 任务之间的采样概率指数级的改变。主要依赖 `hparam` 中的超参：`multiproblem_schedule_threshold` 和 `multiproblem_schedule_max_examples` ，其中：

    * Primary-Task 在train-steps = 0时，有接近100%的采样概率。随着 `train_steps` 的增大，概率开始指数衰减，并且当`train_steps`增大至 `multiproblem_schedule_max_examples` 时，概率跌至 `multiproblem_schedule_threshold` 后持续不变；
    * 剩下的任务，采样概率均分 `1 - P(primary_task)`；

* **pretain:** 预训练模式，主要依赖参数：`multiproblem_schedule_max_examples`，其中：

    * 当 `train_steps < multiproblem_schedule_max_examples` 时，primary-task以 100%的概率进行数据采样；
    * 否则，primary-task 不再参与采样，其他 task 均分 100% 的采样概率；

除了上面三种正式定义的数据混合策略之外，代码中还提供**第四种**数据混合方式，这种方式本质上也是 **consant** 的，通过定义超参中 `multiproblem_per_task_threshold`，
来自定义每个任务被采样的概率。例如：

```python
multiproblem_per_task_threshold = "320,160,10,10"
```

#### 3.1.2 其他注意杂项：

1. **任务ID：** 在multi-problem中，每个任务的 ID 定义如下。在需要用到 `task_id` 的时候(例如在inference阶段，可以通过在`decode_hparam='multiproblem_task_id=xxx'`来指定那个任务进行inference)，可以下面的函数进行反推。

```python
  def update_task_ids(self, encoder_vocab_size):
    """Generate task_ids for each problem.

    These ids correspond to the index of the task in the task_list.

    Args:
      encoder_vocab_size: the size of the vocab which is used to compute
        the index offset.
    """
    for idx, task in enumerate(self.task_list):
      task.set_task_id(idx + encoder_vocab_size)
      tf.logging.info("Task %d (%s) has id %d." %
                      (idx, task.name, task.task_id))
```

### 3.2 MultiProblemV2:
    
> 代码定义：`tensor2tensor/data_generators/multi_problem_v2.py`

跟上面第一版的 multi-problem相比，multi-problem-v2 在总体的功能上并没有太大的区别。只是在不同任务的数据混合和调度方面更加『干练简洁』，省略了大量超参的定义和使用，而是仅仅
使用了 `schedule` 这一字段；

#### 3.2.1 Schedule 字段：

`Schedule`阶段时 MultiProblemV2 类对象初始化时需要提供的参数之一。其是一个包含三元素的tuple字段，分别是：

* `interpolation`: 插值方式，指代了从当前策略到下一种策略演进时的变化过程。包含 `step` 和 `linear` 两种方式；
* `steps`: 一个长度为 $$N$$ 的数组，其中 $$N$$ 表示了策略的次数，而 $$steps[i]$$ 则表示了策略 $$i$$ 开始于哪一个 steps；
* `pmf`: 每个任务的概率质量函数列表，一个形状为 $$[N, M]$$ 的数组，其中 $$M$$ 一般和任务的数量相等，$$pmf[i]$$ 表示了在策略 $$i$$ 时，不同任务采样的概率密度函数；

在上面中可能会有一些术语比较生涩难懂，因此我这里直接举一些简单的例子来看看:

```python
schedule = 'steps', (0, 100, 200),   [[0.1, 0.9], [0,2, 0.8], [0.0, 1.0]]
```

上面的这个schedule表示了：插值方式是 steps：
* 在 $$0 - 100$$ steps 时，任务 0 按照概率值 $$0.1$$ 进行采样，任务 1 按照概率值 $$0.9$$ 进行采样；
* 在 $$100 - 200$$ steps 时，任务 0 按照概率值 $$0.2$$ 进行采样，任务 1 按照概率值 $$0.8$$ 进行采样；
* 在 $$200$$ steps之后的采样，全部采样任务1；

#### 3.2.2 Schedule 的 encode 和 decode：

为了方便，`multiproblem_v2` 中提供了：

* 从 schedule 的tuple形式，将其encode成字符串的形式；
* 根据字符串形式，将其decode成对应的 schedule tuple 形式；

其中，对应的字符串格式的规则如下：
```
"""
(1) 'step @0 0.7, 0.3': Sample from problem 0 w.p. 0.7 and problem 1 w.p. 0.3
    for the entirety of training. Since there is only one point, the choice of
    interpolation method and global_step does not matter.

(2) 'step @0 1.0 0.0 @100 0.0 1.0': Train on problem 0 for the first 100 steps
    then train on problem 1 for the rest of training.

(3) 'step @0 0.5 0.5 0.0 @100 1.0 0.0 0.0': Pretrain on problems 0 and 1 for the
    first 100 steps then fine tune on problem 2 for the rest of training.

(4) 'linear @0 1.0 0.0 @100 0.0 1.0' Linear transition from training on problem
    0 to problem 1 over 100 steps, then train on problem 1 for the rest of
    training.

(5) 'linear @0 1.0 0.0 @100 0.9 0.1  @200 0.4 0.6  @300 0.0 1.0': Approximate
    inverse exponential decay from problem 0 to problem 1 over 300 steps, then
    train on problem 1 for the rest of training.
"""
```

## 0x04. 引用

* [1][Tensor2Tensor - Multi-problem training](https://github.com/tensorflow/tensor2tensor/blob/master/docs/multi_problem.md)
* [2][Github: tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)
* [3][MULTI-TASK SEQUENCE TO SEQUENCE LEARNING](https://arxiv.org/abs/1511.06114)
