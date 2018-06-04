---
layout: post
title: "C++11： Auto and Decltype 详解"
modified:
categories: 语言特性
description:
tags: [c++11]
comments: false
share:
date: 2015-04-03T04:39:00-04:00
---

本文主要是翻译、整理以及补充 Reference 中的参考资料，如果错误，请参考Reference文档；

## Reference

* [C++ auto and decltype Explained](http://thbecker.net/articles/auto_and_decltype/section_01.html)
* [如何评价 C++ 11 auto 关键字？](https://www.zhihu.com/question/35517805)
* [C++ Primmer - auto and decltype]()

C++11新引用的`auto`和`decltype`关键字，都是方便我们在某种情况下，利用编辑器的自动推导功能来推算出具体的类型。但是两者具体有哪些异同，本文
做一个详细的总结。

## Auto关键字

### 基础知识
编程时常常需要把表达式的值赋给变量，这就要求在声明变量的时候清楚的知道表达式的类型。然而要做到这一点并非容易，有时候根本做不到。
为了解决这一问题，C++11新标准引入了`auto`类型说明符，利用它就可以让编译器替我们去分析表达式所属的类型。
值得注意的一点是，**`auto`关键字让编译器通过初始值推算变量的类型，显然，`auto`定义的变量必须有初始值**

下面先列举一些`auto`关键字常用用法：

`auto`令冗长的变量定义变得简洁、节省了需要编码的功夫，尤其是容器的`iterator`，这也是经常举的例子：

{% highlight c++ %}
vector<int> v;
vector<int>::iterator itr = v.begin();  // case 1 without `auto`
auto itr2 = v.begin();                  // case 2 with `auto` keyword
{% endhighlight %}

`auto`配合 lambda 使用很方便，对于 lambda 来说，其是一个 callable object 类型，每一个类型都是独一无二的，
这个类型只有编译器知道，于是你可以：

{% highlight c++ %}
// use `auto` keyword to refer the type of the callable object;
auto closure = [](const int&, const int&) {}
{% endhighlight %}

对于上面，虽然你也可以使用`std::function`来存储，但是这远远没有`auto`来的优雅；
与此同时，在C++14中，泛型lambda也允许了参数类型可以为`auto`类型，如

{% highlight c++ %}
// use `auto` keyword to refer the type of the callable object;
auto closure = [](auto x, auto y) {return x * y;}
{% endhighlight %}

利用`auto`作为占位符，配合`decltype`，来实现返回类型的自动推导：(对于`c++14`，已经可以不需要`decltype`，直接使用`auto`来完成函数返回类型的自动推导了。)
{% highlight c++ %}
template<class T, class U>
auto mul(T x, T y) -> decltype(x * y)
{
    return x * y;
}
{% endhighlight %}

### 特殊用法

编译器推断出来的 auto 类型有时候和初始值的类型并不完全一样。**编译器会适当地改变结构的类型使其更符合初始化规则**

>  当 `auto` 关键字从初始表达式中推断变量的命名时，其有如下的一些特殊规则：
>  1. 如果初始化表达式是一个引用(reference), 则这个引用将会被忽略；
>  2. 如果在上一步之后，表达式中还有一个 top-level 的 `const` 或者 `volatile` 关键字，也会被忽略
>  3. 以上的两个规则，在某些特殊情况下，会有一定的修订(amendments)

针对上面的第一条，可以理解为`auto`是通过初始化这个行为来推导出变量类型的。而当引用被用作初始值时，真正参与初始化
的其实是引用对象的值，因此此时编译器会以引用对象的类型作为`auto`的类型；

例子：
{% highlight c++ %}
int x = 0;
const int& crx = x;
x = 42;
assert(crx == 42 && x == 42);   // since crx is a reference to x; therefore crx = x = 42;
auto something = crx;           // Follow the rule above, `something` here is type of `int`
something = 43;                 // something has no `const` feature;
assert(crx == 42 && something == 43); // Since crx is not a reference;
{% endhighlight %}

如果想要显式的将引用和`const`赋给变量，可以这样写：
{% highlight c++ %}
const int ci = 10;
const auto& k1 = c1;
{% endhighlight %}

#### **特殊修订**

在上面的规则3中，我们提到在特殊的情况下，1，2会有一些修订，接下来将介绍这两种特殊的情况；

##### First Amendment: 
{% highlight c++ %}
const int c = 0;
auto& rc = c;
rc = 44;    // error; const qualifier was not removed;
{% endhighlight %}

如果我们严格遵循上面的两条规则，则应当是`const`关键字会被移除，然后再添加上引用；但是
这样做会令`rc`变成了一个const变量的引用，从而导致`rc`可以修改这个const变量的值。因此，
`auto`在此处保留了这个const得关键字；

##### Second Amendment: 
当`auto`关键字遇上右值引用的时候，会有一些特殊的规则：
> * 如果`auto`的初始表达式是一个lvalue，则被`&&`装饰的auto关键字会先执行正常的推导，然后再添加一个左值引用在此；
> * 如果`auto`的初始表达式是一个rvalue，则被`&&`装饰的auto关键字仅仅会执行正常的推导；

例子：

{% highlight c++ %}
int i = 42;
// 此处i是一个lvalue，因此正常的推导下，auto应该为int，再添加一个左值引用，则最后`ri_1`
// 的值应该是 int&& &, 根据引用坍塌的规则，`ri_1` 的类型是 `int&`
auto&& ri_1 = i; 
// 此处42是一个rvalue，因此正常的推导下，auto应该为int; 因此最后`ri_2`的值是`int&&`
auto&& ri_2 = 42;
{% endhighlight %}


#### **总结**

上面的规则总结了下来，就是C++11希望大家在使用`auto`关键字的时候，有 const/reference/指针 的时候，
不要把他们隐藏在auto之下，而是要显式的使用：
{% highlight c++ %}
const auto* euc = GetCurrentEndUserCredential(); //*
{% endhighlight %}

**核心还是为读者考虑的：1.让他们对应该注意的东西引起足够的注意；2.不要让他们在不重要的东西上浪费太多精力。**

## Decltype 关键字
结合上面的auto自动推导的用法，有时候我们希望从表达式的类型推断出要定的变量类型，但是**不想用该表达式的值来初始化变量**。 为了满足这一要求，C++11 新标准引入了第二种类型说明符 `decltype`。它的作用是选择并返回操作数的数据类型，在此过程中，编译器分析表达式并得到他的类型，**但是却不实际计算表达式的值。**

从上面的一段话我们已经可以大概看的出`auto`和`decltype`的区别了，**当我们需要某个表达式的返回类型，但是又不想实际只想它的时候，就用`decltype`**。

### 常见用法

{% highlight c++ %}
// 推导表达式类型
int i = 4;
decltype(i) a;  // 推导结果为int，a的类型为int

// 与using/typedef 结合，用于定义新的类型
using size_t = decltype(sizeof(0));//sizeof(a)的返回值为size_t类型
using ptrdiff_t = decltype((int*)0 - (int*)0);
using nullptr_t = decltype(nullptr);
vector<int >vec;
typedef decltype(vec.begin()) vectype;
for (vectype i = vec.begin; i != vec.end(); i++)
{
    //...
}

// 泛型编程中结合auto，用于跟踪函数的返回值
template <typename _Tx, typename _Ty>
auto multiply(_Tx x, _Ty y)->decltype(_Tx*_Ty)
{
        return x*y;
}
{% endhighlight %}

### Decltype是怎么进行类型推导的
除了最开始谈到`auto`和`decltype`的使用方式不同之外，两者在类型推导的具体规则也
不太一样，一下我们来聊聊`decltype`类型推导的背后规则：

> 如果推导式 `expr` 是一个原始(plain)，没有括号包裹(unparenthesized)的变量、函数参数、
或者Class member，那么`decltype(expr)`的类型，就是表达式在**代码源码** 中定义的类型；

例子：
{% highlight c++ %}
struct S {
      S(){m_x = 42;}
        int m_x;
};

int x;
const int cx = 42;
const int& crx = x;
const S* p = new S();
//* x is declared as an int: x_type is int.
typedef decltype(x) x_type;
// cx is declared as const int: cx_type is const int.
// but if we use auto, auto will drop the const qualifier
typedef decltype(cx) cx_type;
// Note that p->m_x cannot be assigned to. It is effectively
// constant because p is a pointer to const. But decltype goes
// by the declared type, which is int.
typedef decltype(p->m_x) m_x_type;
{% endhighlight %}

从上面例子中可以看到，如果推导式满足上面的规则，则`decltype`推导的类型就是
类型的源码，**并不会像auto一样，将顶层的引用或者const去掉**

> 如果推导式 `expr` 并不满足上面的规则（plain，unparenthesized...），则令`T` 
表示`expr`的类型。如果：
>  * `expr`是一个左值(lvalue)，那么`decltype(expr) is T&`
>  * `expr`是一个 `xvalue`, 那么 `decltype(expr) is T&&`
>  * `expr`是一个 `prvalue`, 那么 `decltype(expr) is T`

在上面的规则定义中，我们提到了`xvalue`和`prvalue`两个比较陌生的术语。这两个术语都是针对
右值而言的，**一个rvalue如果符合以下条件之一，则其就是一个是`xvalue`, 否则就是`prvalue`**

* A function call where the function's return value is declared as an rvalue reference. An example would be `std::move(x)`.
* A cast to an rvalue reference. An example would be `static_cast<A&&>(a)`.
* A member access of an xvalue. Example: (`static_cast<A&&>(a)).m_x`.

规则听着有些绕，我们还是直接看些[例子](http://thbecker.net/articles/auto_and_decltype/section_07.html)吧：

{% highlight c++ %}
const S foo();
const int& foobar();
std::vector<int> vect = {42, 43};

// foo() is declared as returning const S. The type of foo()
// is const S. Since foo() is a prvalue, decltype does not
// add a reference. Therefore, foo_type is const S.
//
// Note: we had to use the user-defined type S here instead of int,
// because C++ does not allow us to return a basic type as const.
// (Ok, it does allow it, but the const would be ignored.)
//
typedef decltype(foo()) foo_type;

// auto drops the const qualifier: a is an S.
//
auto a = foo();

// The type of foobar() is const int&1, and it is an lvalue. 
// Therefore, decltype adds a reference. By the C++11 reference
// collapsing rules, that makes no difference. Therefore,
// foobar_type is const int&.
//
typedef decltype(foobar()) foobar_type;

// auto drops the reference and the const qualifier: b is
// an int.
//
auto b = foobar();

// The type of vect.begin() is std::vector<int>::iterator.
// Since vect.begin() is a prvalue, no reference
// is added. Therefore, iterator_type is
// std::vector<int>::iterator.
//
typedef decltype(vect.begin()) iterator_type;

// auto also deduces the type as std::vector<int>::iterator,
// so iter has type std::vector<int>::iterator.
//
auto iter = vect.begin();

// std::vector<int>'s operator[] is declared to have return
// type int&. Therefore, the type of the expression vect[0]
// is int&1. Since vect[0] is an lvalue, decltype adds a
// reference. By the C++11 reference collapsing rules,
// that makes no difference. Therefore, first_element has
// type int&.  
//
decltype(vect[0]) first_element = vect[0];

// second_element has type int, because auto drops the reference.
//
auto second_element = vect[1];
{% endhighlight %}
