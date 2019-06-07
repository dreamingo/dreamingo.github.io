---
layout: post
title: "Transformer系列：初探Tensor2Tensor：自定义Problem & 词表"
modified:
categories: 机器学习 
description: "deep-learning transformer attention"
tags: [tensor2tensor deep-learning transformer attention]
comments: true
mathjax: true
share:
date: 2019-06-06T15:15:40+21:12
---

> 本文通过利用 Tensor2Tensor来自定义一个机器翻译问题，以此来对系统中自定义problem、数据、词表的生成等过程有一个深入的了解。

## 0x01. 前言：
在Tensor2Tensor的[github repostory](https://github.com/tensorflow/tensor2tensor)中，阐述了一个实验的训练步骤主要包括：数据生成（DataSet）、设备信息（Device），超参(HyperParam)，算法模型(Model)等若干方面。当我们需要定义一个自己的Problem的时候，可以先从数据生成纬度进行入手，但是，受限于官方及其稀缺的文档和复杂的系统设计，当如果我们要定义一个和官方提供示例不太一样的问题时，会显得比较困难。这个问题在文章[《“变形金刚”为何强大：从模型到代码全面解析Google Tensor2Tensor系统》](https://cloud.tencent.com/developer/article/1153079) 中也指出：当我们要深入的了解 Tensor2Tensor 系统时，会遇到以下的一些问题：

> ...
> 系统支持多任务，任务混杂，导致代码结构比较复杂。在实现的时候，要考虑到整体的结构，所以会存在各种**封装、继承、多态**的实现。可能你只想用其中的一个功能，理解该功能对应的代码，但是却需要排除掉大量的不相关的代码。
> 多层继承和多态也降低了代码的可读性。追溯一个类的某个方法的时候，需要看到其父类的父类的父类。。。这些父类和子类之间的方法又存在着调来调去的关系，同名方法又存在着覆盖的关系，所以要花一些时间来确定当前的方法名到底是调用的的哪个类中的方法。
> ...

但万幸的是，官方源码中的注释、docstring 书写的还是非常详细的，这有助于我们深入的对源码进行了解。 下面的笔记，是基于我们**自定义一个翻译实验**时，所探索出来的一些笔记，如有错误之处，欢迎指出。


## 0x02. 自定义problem：

### 2.1 自定义项目：
下面是自定义问题中项目文件结构：
```python
.
├── __init__.py             # 必须的，用于在注册problem时指示import路径
├── decode_data.sh          # decode 脚本
├── gen_data.sh             # 数据生成脚本
├── run.sh                  # 训练脚本
└── translate_enhi.py       # 自定义problem代码
```

其中值得注意的是，在 `__init__.py` 文件中，需要将自定义的`problem`模块暴露出去，这才能在运行 `t2t-datagen` 等程序时，指定路径后能够成功加载问题。

```python
# __init__.py
from . import tranlate_enhi
```
同时，在项目运行时，以数据生成`gen_data.sh`为例，我们来看看具体需要注意和定义的参数（重点关注 `--t2t_usr_dir` 和 `PROBLEM`的命名）：
```bash
#!/bin/bash
# ge_data.sh

# 官方默认 problem 的名字，为自定义 Problem Class（Class本身默认为大驼峰命名方式） 的小写下划线形态。
PROBLEM="translate_enhi_distinct_vocab"  	
PROJECT_DIR=".."
USR_DIR=$PROJECT_DIR/t2t_project
DATA_DIR=$PROJECT_DIR/data/t2t_data
TMP_DIR=$PROJECT_DIR/data/t2t_tmp_dir

mkdir -p $DATA_DIR $TMP_DIR

t2t-datagen \
    --t2t_usr_dir=$USR_DIR \  		# 重要，指定自定义problem文件的路径，才能使得problem注册成功；
    --data_dir=$DATA_DIR \   		#  数据目录，存储二进制的 tfrecord 数据文件和词表vocabulary文件；
    --tmp_dir=$TMP_DIR \             # 临时目录，存储原始的数据；
    --problem=$PROBLEM   		 # 自定义的问题名字；
```

### 2.2 Problem的定义:
在t2t中，位于 `data_generators.problem.Problem` 是所有后定义 Problem 的基类。其中，以翻译问题(`tranlate_ende_wmt_bpe32k`)为例，起Problem的继承路线是这样的：
```
Problem>>Text2TextProblem>>TranslateProblem>>TranslateEndeWmtBpe32k>> 
```
因此，当我们要实现独立的翻译任务时，正确的做法是定义自己的问题，并继承 `TranslateProblem`。同时，所有的问题（problem）通过在利用 `utils/registry.py` 中的 `@registry.register_problem` 修饰器进行注册，这才能在使用的时候，通过名字就可以直接调用对应的problem。

```python
# translate_enhi.py
@registry.register_problem
class TranslateEnhiDistinctVocab(translate.TranslateProblem):
    """ 英语-印地语 翻译项目，源语言和目标语言不共享词表 """
    ...
```

## 0x03. 数据生成：
在自定义的problem中，最主要的功能就是数据的处理过程，其中主体流程主要包括：

1. `bin/t2t-datagen.py` 中，调用对应注册problem的 `generate_data(data_dir, tmp_dir)` 函数，作为数据处理的总入口；
2. 对原始数据进行处理，包括下载、解压、清洗的步骤，存放于 `tmp_dir`文件夹中；
3. 根据配置生成二进制的 tfrecord 训练(train)、校验(eval)数据集，存放在 `data_dir` 目录下；
4. 生成额外的文件（如词表），存放在 `data_dir` 目录下；

接下来，我们从顶级入口 `generate_data()` 入手，去分析整个数据生成的过程。正如最开始谈到，不同的problem之间，根据自身的实际需要，对整个继承链路的不同数据函数进行了重载、复写。但是总体大框架函数作用还是如下图所示：


<figure>
	<a href=""><img src="/assets/images/tensor2tensor/func.jpg" alt="" width="600" heigh="600"></a>
    <figcaption><a href="" title="">Tensor2Tensor 数据生成重要接口关系</a></figcaption>
</figure>

我们接下来，直接从源码的角度来对数据进行分析：

##### 1. **顶层入口：`def generate_date(self, data_dir, tmp_dir)：`**

在这里，我们以 `Text2TextProblem` 中的具体实现为参考：

```python
  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """
    步骤：
        1. 确定一系列需要生成的训练/校验文件名列表；
        2. 根据这些文件名，调用 `generate_encoded_samples` 生成对应的文件；
        3. shuffle
    核心函数： generate_encoded_samples(data_dir, tmp_dir, split)
    """

    # dict中的values都是一个callable函数。用于根据传入参数，返回对应的文件路径。
    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    # 作用：确定每个split(train，eval)的文件路径列表集合：
    # 根据每个split，例如 `train` & `evaluation`，以及每个split需要切割的份数（shards）,
    # 对应的文件列表。例如 `{problem_name}-train-00043-of-00100`
    split_paths = [(split["split"], filepath_fns[split["split"]](
        data_dir, split["shards"], shuffled=self.already_shuffled))
                   for split in self.dataset_splits]
    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    # 如果是 self.is_generate_per_split, 则每个 split (train, eval) 都调用一次
    # generate_encoded_samples 函数，用于生成对应的文件。
    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(
            self.generate_encoded_samples(data_dir, tmp_dir, split), paths)
    # 否则的话，则只生成 problem.DatasetSplit.TRAIN 生成。 
    else:
      generator_utils.generate_files(
          self.generate_encoded_samples(
              data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

    generator_utils.shuffle_dataset(all_paths, extra_fn=self._pack_fn())
```

##### 2. **生成编码后的数据：`def generate_encoded_samples(data_dir, tmp_dir, dataset_split)`:** 
这里直接以 `Text2TextProblem` 中的函数定义进行分析：该函数返回一个 Python Generator，生成编码后的样本数据（以文本问题为例，就是对文本进行Tokenize ID化）；可以看到，程序中的核心步骤有：

* 调用 `generate_samples` 函数，获得一个迭代器，用于生成数据；
 * 获取词表的encoder（注意的是，对于subword-encoder的生成，函数里面实现同样会先调用 `self.generate_samples`，生成一个迭代器，遍历train数据集，来生成词表，这一点在语言模型中很常见，一般先遍历一次文件，生成词表，然后再多次遍历文件来迭代训练）
* 根据迭代器 `generator` 和 `encoder`，调用函数 `text2text_generate_encoded` 函数进行生成encoded数据，返回的同样是一个迭代器。

 其中**值得注意：**
* 一般的子类会重载 `self.generate_samples` 函数，yield 方式来生成数据。
*   这里指返回了一个encoder，**因此 `Text2TextProblem` 中，默认 input 和 target sentence 是共用一个词表 encoder **，如果例如翻译问题中，两种语言想用不同的词表（例如两种语言的词法、字母表都完全不一致），则需要重载这个函数，来修改 encoder 的获取方式。
* `text2text_generate_encoded` 函数是可以分别接收 `input_encoder` 和 `target_encoder` 的，在下面的源码中忽略了 target encoder，则会令 input 和 target 共用同一份词表；

```python
  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
   	"""
	Yields:
	---
		{"intpus": encoded data, "outputs": encoded data}
	"""
    # ....
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
	# 利用 encoder，对generator生成的数据进行编码；
    return text2text_generate_encoded(generator, encoder,
                                      has_inputs=self.has_inputs,
                                      inputs_prefix=self.inputs_prefix,
                                      targets_prefix=self.targets_prefix)
```

##### 3. **实际的数据生成接口：`def generate_samples(self, data_dir, tmp_dir, dataset_split)`:**

在这里，我们参考 `TranslateProblem.generate_samples`  的定义来分析；源码中的这个函数说实话，看起来比较绕，因为实现了比较多『花里胡俏』的功能，例如根据url下载数据文件、解压、清洗等工作。

实际上，值得关注的只有最后一句话：`return custom_iterator(input_files, target_files)`,  实际上返回的就是一个zipdict，将input文件中的每行 和 target_files 中的每行zip起来，实现返回一个 `{"inputs": txtline, "output": txtline}` 的功能；

```python
  def generate_samples(
      self,
      data_dir,
      tmp_dir,
      dataset_split,
      custom_iterator=text_problems.text2text_txt_iterator):
	"""
	Yields:
	---
		{"inputs": str_txt, "targets": str_txt}
	"""
    # 在官方提供的例子中，dataset中包含了 [url, (filename1, filename2)] 两个元素，
    # 主要做了一些下载、解压，对filename1, filename2 中的数据进行清新的功能；
    datasets = self.source_data_files(dataset_split)
    tag = "dev"
    datatypes_to_clean = None
    if dataset_split == problem.DatasetSplit.TRAIN:
      tag = "train"
      datatypes_to_clean = self.datatypes_to_clean
    # 实际返回的文件，是清洗好(compiled)的文件，例如`/tmp/t2t_datagen/translate_ende_wmt32k-compile-dev/train.lang1/lang2` 的中间文件
	# compiled的功能包括数据清洗、解压、多个文件合并成一个等子功能；
    data_path = compile_data(
        tmp_dir, datasets, "%s-compiled-%s" % (self.name, tag),
        datatypes_to_clean=datatypes_to_clean)

    return custom_iterator(data_path + ".lang1", data_path + ".lang2")
```

---

## 0x04. 词表生成：
对于一个翻译任务，词表（Vocabulary）以及由其生成的 TextEncoder 对象是一个重要的组成部分，这涉及到如何对原始符号到 id 的编码。其中，T2T的`text_problems` 所支持的词表类型有：

```python
class VocabType(object):
  """Available text vocabularies."""
  CHARACTER = "character"
  SUBWORD = "subwords"
  TOKEN = "tokens"
```
在机器翻译任务中，目前主要用到的还是基于 subwords 编码方式的词表，这种方式一来可以有效的减少词表的大小，二来可以有效的令模型对罕见词（人名、专有名词、错别字等）解决OOV（Out of vocabulary）的问题。T2T里面构造subword的算法，和传统 [Sennrich](https://arxiv.org/abs/1508.07909) 基于迭代merge次数来控制词表大小的方式不太一样，而是使用二分查找的方式，通过搜索最优的minimum token count值来逼近预先设置的词汇表的大小。

在上面函数  `generate_encoded_samples` 函数中，可以主要是看到调用 `get_or_create_vocab` 函数来获得/创建词表encoder的。那么我们接下来分析下这个函数：

```python
  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
	""" 根据配置，从本地读入/创建一个 TextEncoder """
    if self.vocab_type == VocabType.CHARACTER:
      encoder = text_encoder.ByteTextEncoder()
    elif self.vocab_type == VocabType.SUBWORD:
	  # 如果是 force_get 的话，则直接从定义好的 
	  #`self.vocab_filename` 文件中生成 SubwordTextEncoder
	  if force_get:
        vocab_filepath = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
      else:
        other_problem = self.use_vocab_from_other_problem
        if other_problem:
          return other_problem.get_or_create_vocab(data_dir, tmp_dir, force_get)
		# 核心： 根据data_dir, 定义的词表大小等配置，和生成词表数据generator，
		# 生成一个 subword词表（写入到 vocab_filename） 和 对应的SubwordEncoder。
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir, self.vocab_filename, self.approx_vocab_size,
            self.generate_text_for_vocab(data_dir, tmp_dir),
            max_subtoken_length=self.max_subtoken_length,
            reserved_tokens=(
                text_encoder.RESERVED_TOKENS + self.additional_reserved_tokens))
    elif self.vocab_type == VocabType.TOKEN:
      vocab_filename = os.path.join(data_dir, self.vocab_filename)
      encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                              replace_oov=self.oov_token)
    else:
      raise ValueError(
          "Unrecognized VocabType: %s" % str(self.vocab_type))
    return encoder
```

在上面的这个函数中，我们可以看到**两个核心**的信息：
1. 如果以后项目中已经有预定义好的词表（例如通过BPE切割好vocabulary文件），则可以直接通过 `text_encoder.SubwordTextEncoder` 来创建一个encoder；
2. 如果项目中还是希望利用 T2T 生成词表以及对应的 encoder，则可以根据自己实际需要（不同的语言用不同的encoder），来实际调用函数 `generator_utils.get_or_generate_vocab_inner`

当我们需要自定义词表的时候，还需要修改重载  `feature_encoders` 函数。该函数主要用于在 eval/decode 阶段，指示模型获取正确的decoder，以此来对输入文本的正确编码；这里，我贴出一份修改后支持 input 和 targets 支持不同 encoder 的函数：

```python
    def feature_encoders(self,data_dir):
        """ **Overwrite function**: 文本编码器
        重载该函数，以实现针对源语言和目标语言使用不同编码器的效果
        Returns:
        ---
            {"inputs": input_encoder, "targets": target_encoder}
        """
        input_encoder = text_encoder.SubwordTextEncoder(
            self._input_vocab_filename)
        target_encoder = text_encoder.SubwordTextEncoder(
            self._target_vocab_filename)
        return {"inputs": input_encoder, "targets": target_encoder}
```

## 0x05. 引用

* [1][“变形金刚”为何强大：从模型到代码全面解析Google Tensor2Tensor系统](https://cloud.tencent.com/developer/article/1153079)
* [2][Tensor2Tensor for Neural Machine Translation](https://arxiv.org/pdf/1803.07416.pdf)
* [3][Github:tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)
