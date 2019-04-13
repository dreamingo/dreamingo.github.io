---
layout: post
title: "RNN系列：Neural Machine Translation with Atttion - Tensorflow 示例分析"
modified:
categories: 机器学习 
description: "deep-learning rnn seq2seq attention"
tags: [deep-learning rnn source-code seq2seq attention]
comments: true
mathjax: true
share:
date: 2019-04-13T18:40:28+21:12
---

> 本文是 RNN 系列的第四篇文章，主要对Tensorflow中，关于Seq2Seq with attention 的一个示例代码进行分析。旨在从实际项目中对 RNN 系列有更深入的了解。

## 0x01. 引言

本文主要是对Tensorflow 2.0 官方提供的一个利用
Seq2Seq with Attention 进行机器翻译(Neural Machine Translation)的示例代码
([Google Colab地址](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/sequences/nmt_with_attention.ipynb))
进行分析和添加自己的注释理解，进一步加强对RNN系列的深入了解。

下面的示例分析中，并非选取所有的代码细节，具体请参考引用链接；

## 0x02. 数据处理

第一部分中，我们首先对数据的处理部分代码进行分析，主要包括：
1. 文本清洗和处理(标点符号的处理和头尾标志的插入)
2. 将文本 Tokenize ID 化 和 padding 到固定长度；
3. 利用 `tf.data.DataSet` 读入并分批化(batch)

{% highlight python %}
# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """
    将字符串中的某些特殊特殊字符转化，例如去除音调（accents）
    主要是针对西班牙语中一些特殊的字符进行处理
        such as: á to a
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    """ 句子预处理
    包括如下步骤：
        1. 特殊的标点符号前后 pad 上空格；
        2. 多余的空格压缩；
        3. 其他的非正常符号一律用空格代替；
        4. 句子前后添加上 <start>, <end> 标志
    eg:
        input: May I borrow this book?
        output: <start> may i borrow this book ? <end>
    """
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    
    # Only pad the punctuaton ?.!,¿
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    # 压缩多余的空格符号
    w = re.sub(r'[" "]+', " ", w)

    # 处理以下符号，其他的全部替代成空格（这里处理的略有些粗暴）
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # 在首尾添加上 <start> <end>，便于模型知道句子的开始和结束
    w = '<start> ' + w + ' <end>'
    return w


def tokenize(lang):
  """ 将输入句子集合 tokenize （ID化），并pad到固定长度；
  Args:
    lang: List[Str], 输入的句子集合，列表中的每个元素是一个句子；
  Returns:
    tensor: 2D-Tensor, with shape [size_lang, max_lenght],
        ID 化 & padding 后的数据。例如：
        [[1, 3, 2, 4, 0, 0, 0], [1, 7, 9, 4, 0, 0, 0]]
    lang_tokenizer: Tokenizer, 用于将text转化为ID，or ID 反推回Text；
  """
  # `filter=''` 表示保留所有的单词, 并且默认是以空格分割的；
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  # Tokenizer ID化，例如：[[a b c], [b, c d]] to tensor vector [[0,1,2],[1,2,3]]
  tensor = lang_tokenizer.texts_to_sequences(lang)

  # Pad all then vector to the max length(post way), with default id 0;
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
  return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    """ 读入数据并处理 """
    # creating cleaned input, output pairs    
    targ_lang, inp_lang = create_dataset(path, num_examples)
    # Input tensor, and the tokenizer object, to convert text to id or
    # revert id to text
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

# 利用 tf.data.DataSet 读入处理好的处理，并根据配置 batch 化：

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
# 可以通过这种方式来获取一个批次的数据：example_input_batch, example_target_batch = next(iter(dataset))
# 其中大小为：(64, input_max_len), (64, output_max_len)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

{% endhighlight %}


## 0x03. Encoder, Decoder & Attention Layer

这一部分主要展示在 Tensorflow 2.0  中如何利用`tf.keras`来实现 `Encoder` 和 `Decoder` 和 `Attention Layer`。其中值得注意的是，下面中公式中的某些符号，和原文会有些出入（因为我发现原文中的符号和实际代码实现有些不符，容易引起混淆。），同时代码中也会根据个人的命名习惯，对一些参数变量的命名进行修改；

在开始之前，同样的我们先对出现的符号作统一的定义：（源码中使用的是Bahdanau Attention，具体可以参考[上一篇博文](http://dreamingo.github.io/2019/04/seq2seq/)）

* $$\mathbf{\bar{h}_s}$$: 表示在Encoder中，timestep $$s$$ 时所产生的hidden-state
* $$\mathbf{h_t}$$: 表示在Decoder 阶段中，第 $$t$$ 步时所产生的 hidden-state。其中，$$\mathbf{h_t} = GRU(concat[[\mathbf{y_{t-1}}, \mathbf{c}_t], \mathbf{h_{t-1}}])$$；值得注意的是，
Decoder的每步输入 $$x_t = concat(\mathbf{y_{t-1}}, \mathbf{c_t})$$
* $$\alpha_{ts}$$: Attention Weights, 定义为： $$\alpha_{ts} = \frac{exp(score(\mathbf{h_t}, \mathbf{\bar{h}_{s-1}}))}{\sum_{s'=1}^{S}exp(score(\mathbf{h_t}, \mathbf{\bar{h}_{s-1}}))}$$

* $$\mathbf{c}_t$$: 输出 $$t$$ 对应的context向量，定义为: $$\mathbf{c}_t = \sum_{s}\alpha_{ts}\bar{h}_s$$
* $$score(h_t, \bar{h}_{s-1})$$: Bahdanau additive style attention score: $$score(h_t, \bar{h}_{s-1}) = v_a^Ttanh(W_1h_t + W_2\bar{h}_{s-1})$$ \_


{% highlight python %}
class Encoder(tf.keras.Model):
    """ 基于GRU 的Encoder 模型
      vocab_size:       Int, 输入词表大小；用于初始化 embedding 层；
      embedding_dim:    Int, Embedding 向量的大小
      enc_units:        Int, Encoder hidden units size
      batch_sz:         Int, batch size
    """
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        """ Call 接口，直接运行完 `max_input_len` 次调用
        Args:
            x:  2D-Tensor, with Shape [batch, max_input_length]，输入序列
            hidden: 2D-Tensor, with shape [batch, hidden_size], Encoder初始化的hidden-state。
        Returns:
            output: Tensor, with shape [batch, max_input_length, hidden_size]
                Encoder阶段中每一步 s 输出的hidden-state： \bar{h}_s 集合；
                主要用于计算 attentions；
            state:  Tensor, with shape [batch, hidden_size]
                Encoder最后一个输出的hiddent-state；一般用于Decoder第一步的初始化hidden-state
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        """ 初始化hidden-state
        Returns:
            hidden: 2D-Tensor, with shape [batch, hidden_size]
        """
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.Model):
    """ Attention Model
    score = V.(tanh(W1.({h}_s)) + W2.(\bar{H}))
    Args:
        W1:  Tensor, with shape [units, hidden_size]
        W2:  Tensor, with shape [units, hidden_size]
        V:   Tensor, with shape [hidden_size, 1]
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

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

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_input_length, 1)
        # axis = 1 是因为归一化时，用的 max_input_length 这一纬度进行归一化 
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    """ 基于GRU的decoder，内嵌Attention weight
    Args:
        batch_sz:         Int, batch size
        dec_units:        Int, Decoder hidden state units size;
        embedding:        Layer, embedding Layer, with shape(vocab_size, embedding_dim)
        fc:               Layer, with shape [hidden_size, vocab_size], 主要用于生成y
    """
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # 主要用于利用h_t生成y_t
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        """ timestep t 每调用一次，生成一个y_t
        Args:
            x:  Tensor, with shape: [batch_sz, vocab_size]，
                输入的x，是t-1阶段预测得到的y值；
            hidden: Tensor, with shape [batch_sz, hidden_size],
                t-1 阶段产生的 hidden-state，主要用于传给Attention层生成context-vector
            enc_output: Tensor, with shape [batch_sz, max_input_len, hidden_size]
                encoder阶段输出的所有hidden-state，主要用于传给Attention层生成context-vector
        Returns:
            x:  Tensor, with shape: [batch_sz, vocab_size]，
                当前阶段输出的y，同时也是下一阶段的输入x；
            state: Tensor, with shape: [batch_sz, hidden_size]
                当前阶段的hidden-state，主要用于喂给下一阶段；
            attention_weights: Tensor, with shape: [batch_sz, max_input_len, 1]
                权重，用于绘图；
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # 可以看到，在这里将上一阶段的输出 [y_{t-1} , c_t] 作为 x， 输入到gru中；
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        # 计算当前的输入y，同时也是下一阶段的输入x
        x = self.fc(output)
        return x, state, attention_weights
{% endhighlight %}

## 0x04. Train & Test

在训练阶段，值得注意的有以下几个点：
1. 对于一个batch的数据，需要根据该batch的输出最大长度 `max_output_len` 来循环，loss则在每步timestep中不断的累加。
2. 在Decoder阶段$$t$$，并不会真的拿上一阶段$$t-1$$的输出 $$y_{t-1}$$作为输入，而是拿真实标记group-truth $$grouptruth(y_{t-1})$$作为输入。在引用文章中，这种技巧叫做：`teacher forcing`。
这应该意味着在训练过程中，每次预测会强行用真实标记进行矫正，有点类似于一个老师每次强迫你矫正错误；
3. 在decoder的第一步中，因为没有上一步的输出，因此 $$x$$ 一般用 `<start>` 的id代替；

{% highlight python %}
@tf.function
def train_step(inp, targ, enc_hidden):
  """
  Args:
    inp: Tensor, with Shape[batch_size, max_input_length]
    targ: Tensor, with Shape [batch_size, max_output_length];
    enc_hidden: Tensor, with Shape [batch_size, hidden-size]
        用于初始化encoder的hidden state
  """
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    # decoer 利用encoder最后一个hidden-state，作为第一步的初始化hidden-state；
    dec_hidden = enc_hidden

    # 第一步中的decoder输入x，用`<start>`的id代替
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # dec_hidden 一直被替代，表示上一阶段的hidden-state
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      # Calucate the loss for the `t` word in this batch;
      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      # Do not use the last prediction, but the real target word as next input;
      dec_input = tf.expand_dims(targ[:, t], 1)

    # divide the max_output_length;
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
{% endhighlight %}

而在预测过程，则有以下几个注意的点：
1. 预测的句子，需要根据训练时同样的数据预处理阶段（清洗、tokenize等）
2. 下一步预测的输入，在这里用的就是上一步的输出 $$y_{t-1}$$了。不会再像training阶段使用teacher forcing 技巧
3. 如果decoder预测的单词为 `<end>`，则结束预测并返回；

{% highlight python %}
def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot
{% endhighlight %}

