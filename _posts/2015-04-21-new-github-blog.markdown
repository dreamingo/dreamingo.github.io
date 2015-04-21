---
layout: post
title: "使用jekyll于github上搭建新博客"
modified:
categories: Geek折腾成果 
description: 依靠github page搭建的新博客
tags: [github,jekyll,markdown blog]
image:
  feature: abstract-2.jpg
  credit:
  creditlink:
comments: true
share:
date: 2015-04-21T13:54:50+08:00
---

在这之前,和同学合租了一个香港的VPS,在上面利用`Python-django`搭建了自己海贼王风格的技术博客，用于记录自己点滴的技术成果及阅读总结。因为本人是个海贼王迷，该博客是大三那年心血来潮搭建的，其中后端逻辑的处理，前端构建，海贼王素材收集，P图等，都是当时辛辛苦苦完成的。那是一个令我相当满意和自豪的博客。

无奈天有不测之风云，于数天前,可恶的VPS供应商(当时我们贪便宜在淘宝联系的卖家)告知服务器遭受攻击，内容全部被清空。这真是令我欲哭无泪。幸好一些当年的笔记的`markdown`草稿还在本地留下。才不至于损失惨重。

吸取了上次的教训，也感觉无力再折腾了。于是就利用`github page`提供可靠的服务，决定在上面利用`jekyll`搭建新的博客。

<figure>
	<a href=""><img src="{{site.url}}/images/new-blog/old-blog-2.png" alt=""></a>
	<figcaption><a href="" title="My old blog">旧博客(www.dreamingo.com:9999) - 遗照</a>.</figcaption>
</figure>

### 为什么使用`Github` & `Jekyll`

`Jekyll`[官方document](http://jekyllrb.com/)上的描述就是:

> _Tranform your plain Text into static websites and blogs_ 
> Jekyll is a simple, blog-aware, static site generator. It takes a template directory containing raw text files in 
various formats, runs it through a converter (like Markdown) and our `Liquid` renderer, and spits out a complete, 
ready-to-publish static website suitable for serving with your favorite web server. 
`Jekyll` also happens to be the engine behind `GitHub Pages`, which means you can use Jekyll to host your project’s page, blog, or website from GitHub’s servers **for free.**

当然，静态网页的缺点也有有的，例如：

+ 生成的是静态网页，无法动态加载。例如需要外部的服务如评论，只能使用
`disquz`，多说等的外部插件。
+ 仅仅适合小型网站，不适合大中形网站。
+ 没有数据库以及服务端的逻辑。

<!--more-->

### 基于Github Page安装Jekyll

强烈建议在您的电脑上安装Jekyll，以便在把博客更新到GitHub Pages 仓库前，能够先预览一下网站，找出内容上可能的错误或者因程序出错引起的编
译错误等问题。

幸运的是，可以通过使用 the GitHub Pages Gem 以及 GitHub Pages 的依赖组件系统，我们可以很简单的在电脑上安装Jekyll 并很
大程度上匹配`GitHub Pages`的设置。

安装Jekyll 能简单到什么程度？其实，你只需完成下面的几个步骤就OK了：

#### 安装Ruby

1. Jekyll 运行在`Ruby` 上，所以，你懂的！如果你使用Mac，你电脑分分钟已经安装了Ruby。打开终端，运行一下命令即可知道有没有安装Ruby，以及Ruby 的版本：`ruby --version`. GitHub Pages 需要Ruby 的版本至少是 `1.9.3` 或者 `2.0.0`。如果你已经满足上面Ruby 的条件，可以直接跳到第二个步骤了，
否则，看看[这些文章](https://www.ruby-lang.org/en/downloads/)去安装Ruby 吧!

#### 安装Bundler

Bundler 是Ruby 上一个组件包版本管理的利器。你想在本地上安装 `GitHub Pages` 的环境，安装`Bundler`就很有必要了！可以通过以下命令完成安装：

{% highlight bash %} 
gem install bundler。
{% endhighlight %} 

值得注意的一点是由于网络原因，从`rubygems.org`下载gem文件速度非常缓慢，甚至间歇性连接失败。所以可以修改gem源为`ruby.taobao.org`.详细的
官方document，可以参考[RubyGems镜像](http://ruby.taobao.org/).

{% highlight bash %} 
{% raw %} 
$ gem sources --remove https://rubygems.org/
$ gem sources -a https://ruby.taobao.org/
$ gem sources -l
\*\*\* CURRENT SOURCES \*\*\*

https://ruby.taobao.org
# 请确保只有 ruby.taobao.org
{% endraw %} 
{% endhighlight %} 


#### 安装Jekyll

你需要在网站的根目录创建一个名为Gemfile的文件，并在里面添加： 
{% highlight bash %} 
source 'http://ruby.taobao.org/'
gem github-pages

{% endhighlight %} 

然后打开终端，切换到上述网站的根目录，输入`bundle install`,进行安装。

#### 运行Jekyll

为了更好的模拟`GitHub Pages`的编译环境，最好使用`Bundler`来运行`Jekyll`。在网站的根目录下运行以下命令来运行`Jekyll`：

{% highlight bash %} 
bundle exec jekyll serve
{% endhighlight %} 

默认的，就可以通过`http://localhost:4000`来访问本地的网站。

#### 更新Jekyll

由于Jekyll 是一个开源项目，而且经常会有更新，影响是，如果GitHub Pages 的编译器更新了jekyll，而我们本地的jekyll 还是在旧的版本，
就很有可能会导致网站在本地的表现与线上Github Pages 网站不一样。
所以，保持Jekyll 与Github Pages 编译器用的一样是超级超级重要的，运行下面的命令就很好的达到目的：
{% highlight bash %} 
bundle update
{% endhighlight %} 

这里更加体现了，使用`github-pages`来安装`Jekyll`是多么的好呀

* * * 

### 新主题HSPTR
本博客使用主题[HPSTR](http://jekyllthemes.org/themes/hpstr/)。并且基于网友[Seeksky的博客](http://jinlaixu.net/)在`HPSTR`上的修改，我再稍作修改。

主要改进的地方：

* 悬浮TOC（在Seeksky的基础上，增加了对三级，四级标题等支持）
* 依然使用`pygments`作为代码高亮的处理。
* 优化资源文件访问过慢的问题(详细请参考[优化Jekyll博客访问慢的问题](http://9leg.com/other/2015/01/15/optimization-of-jekyll-blog-access-slow-problem.html))：
    + 将使用的`google-fonts`用360的公共库useso代替。
    + 将`jquery`等资源镜像换位国内镜像文件。

关于本主题的基础架构，以及`Theme Setup`,可以参考`HPSTR`的[Theme Setup](https://mmistakes.github.io/hpstr-jekyll-theme/theme-setup/)

### [Reference]

+ [使用Jekyll在Github上搭建个人博客（环境搭建）](http://segmentfault.com/a/1190000000406011)
+ [基于GitHub Pages 安装Jekyll](http://blog.ssyog.com/blog/jekyll/install-jekyll-based-on-github-pages.html)
+ [Using Jekyll with Pages](https://help.github.com/articles/using-jekyll-with-pages/)
+ [Jekyll Document](http://jekyllrb.com/docs/home/)
