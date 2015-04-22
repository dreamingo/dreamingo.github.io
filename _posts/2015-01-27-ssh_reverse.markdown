---
layout: post
title: "反向连接访问局域网内机器"
modified:
categories: Linux&Geek
description:
tags: [ssh, 反向代理, linux]
image:
    feature: abstract-10.jpg
    credit:
    creditlink:
comments: true
share:
---

现有实验室机子A一台，宿舍机子B一台，公网服务器C一台。由于实验室访问外网速度巨慢，而且在宿舍工作也是各种方便,希望能够在宿舍远程登陆实验室机子做各种实验。但是由于相对于宿舍机子B来说，实验室是局域网网络，B是无法直接连接A的（但是A可以直接连接B）。因此，我们可以借助一台公网服务器，通过ssh的反向连接代理，访问实验室机子B。

### 基本操作

**将机子A的22端口反向代理到公网服务器C：**

{% highlight bash %}
# 将A的22端口绑定到公网服务器C的2345端口
ssh -NfR 2345:localhost:22 username_of_C@C_ip_address
{% endhighlight %} 

**参数说明：**

详细的参数说明可以参考ssh的[man page](http://linux.die.net/man/1/ssh)

* `-N`: 不执行任何指令
* `-f`: Requests ssh to go to background just before command execution（后台执行）.
* `-R`: 用于建立reverse tunnel

输入服务器C的密码并且绑定成功后，在服务器C中，通过

{% highlight bash %} 
#-p为指定端口参数
ssh username_of_A@localhost -p2345
{% endhighlight %}  

即可登陆到机子A中。

<!--more-->
### 进阶操作

#### **减少操作延时:**

如今，假设我们从宿舍机子B，访问实验室机子A，我们需要：

* 登陆到公网服务器C中；
* 在C中ssh连接实验室机子A，由此操作A。

在这里，我们的控制链是`B->C->A`。如果公网服务器位于国外，或者线路不太稳定，将导致我们的控制链延时非常大。在这里，瓶颈是C；

于是，我们可以充分发挥校园局域网的优势（或者是国内速度比国外服务器速度更快的优势），具体操作如下：

* 从B登陆公网服务器C；
* 在C中`ssh username_of_A@localhost -p2345`登陆到A中。
* 在A中执行命令`ssh -NfR 1234:localhost:22 username_of_B@B_ip_address`
* 在宿舍机子B中，直接执行`ssh username_of_A@localhost -p1234`登陆到A中。


#### **无密码登陆**

这个已经是一个老生长谈的问题，具体操作：

在实验室机子A中产生ssh公钥和私钥

{% highlight bash %}
$ ssh-keygen #...(一直按Enter，最后在~/.ssh/下生成密钥)
$ ls ~/.ssh/id_rsa id_rsa.pub known_hosts
{% endhighlight %}

复制产生的`id_rsa.pub`公钥到外网机子C中，并且将内容拷贝到C的`~/.ssh/authorized_keys`中


#### **autossh**

ssh本身受网络因素影响较大，一旦网络不好随时可能断开。这时候需要内网机子A再次向其“外网”机子发起反向连接请求。这时候，我们可以使用autossh。
autossh提供了断线自动连接功能，同时也可以解决超时重连等问题。

{% highlight bash %} 
sudo apt-get install autossh
autossh -f -M 5678 -NR 1234:localhost:22 username_of_C@C_ip_address
{% endhighlight %} 

比起之前的命令多了`-M 5678`参数。负责通过5678端口监听连接状态，连接有问题时候就会自动重连。如果向实验室内网机子A在重启后自动执行autossh命令，
那么，将其加入daemoon吧（如何写入守护进程请自行查询）。


#### **X Forwarding**

关于X Forwarding的信息，这里就不做介绍了，直说用法。

有时候，当我在实验室的机子画图后，希望能够将结果plot出来，或者我想用eog（ubuntu下图片查看器）查看图片时，又或者我想Forwarding整个matlab的图形界面的这时候，我们可以考虑在ssh中加入X Forwarding， 使其将X信息forward过来。至于X Fowarding的简单配置，可以参考stackexchange上的一个[回答](http://unix.stackexchange.com/questions/12755/how-to-forward-x-over-ssh-from-ubuntu-machine).配置成功后，直接在ssh中加入参数`-X`即可:

{% highlight bash %} 
ssh -X username_of_A@localhost -p2345
{% endhighlight %}


###参考文献

* [SSH反向连接及Autossh](http://7177526.blog.51cto.com/7167526/1391328)
* [How to forward X over SSH from Ubuntu machine?](http://unix.stackexchange.com/questions/12755/how-to-forward-x-over-ssh-from-ubuntu-machine)
