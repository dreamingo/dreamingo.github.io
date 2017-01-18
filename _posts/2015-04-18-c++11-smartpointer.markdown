---
layout: post
title: "C++11： Smart Pointers(智能指针)"
modified:
categories: 语言特性
description:
tags: [c++11]
comments: false
share:
date: 2015-04-18T04:39:00-04:00
---

本文主要是翻译、整理以及补充`Reference`中的参考资料，如果错误，请参考Reference文档；

## Reference

* [LESSON #4: SMART POINTERS](https://mbevin.wordpress.com/2012/11/18/smart-pointers/)
* [簡介C++ 11下的 Smart Pointer 元件 - shared_ptr, weak_ptr, unique_ptr](http://xingularity.info/mindfull/%E7%B0%A1%E4%BB%8Bc-11%E4%B8%8B%E7%9A%84-smart-pointer-%E5%85%83%E4%BB%B6-shared_ptr-weak_ptr-unique_ptr/)
* [Ten C++11 Features Every C++ Developer Should Use](http://www.codeproject.com/Articles/570638/Ten-Cplusplus-Features-Every-Cplusplus-Developer)


C++11一个很巨大的改进在于自从 shared_ptr ,  unique_ptr 和 weak_ptr 在 C++11 被提出来后,程序员就
应该不在手动的管理程序中的内存情况(不需要手动的`delete`或者`free`).

在C++11之前,也有过一个智能指针的类 - `auto_ptr`. 但是,由于其的不安全性,现今已经弃用(deprecated)

要使用上述的所说的这些类,你需要手动的 `#include <memory>` (并且添加 `using namespace std;` 或者添加前缀 `std::` );

## unique_ptr

unique\_ptr 对象简单的持有着一个指针,并且保证这个指针的对象会在 unique_ptr 对象被销毁的时候（例如离开作用域时）同时被销毁.

 unique\_ptr 是一个独享所有权的智能指针，他拥有他所指向的对象，并且 unique\_ptr 对象**不能够被复制构造，也无法进行复制赋值**。
也就是说，不可能同时存在两个  unique\_ptr 指针指向同一个对象。但是可以通过移动构造（move constructor）和移动赋值操作

上述的不可以被复制，在这里举一些例子：

unique\_ptr 本身不能够被复制构造和复制赋值；

{% highlight c++ %} 
class Base {
public:
    Base(unique<int> int_p) {
        this->int_p = int_p;     //Error, 非法使用deleted的复制赋值；
        this->int_p = std::move(int_p); //Correct;
    }
private:
    unique<int> int_p;
};

unique<int> int_p(new int(0));
unique<Base> base_p(int_p);     //Error, 中间涉及到unique_ptr的复制构造；
unique<Base> base_p(std::move(int_p)); //Correct; move consturctor;

{% endhighlight %} 

<!--more-->

但是如果函数返回值为`unique_ptr`的话，这时候并不会触发复制构造，而是触发了移动构造函数（move constructor）

{% highlight c++ %} 
unique<int> func() {
unique<int> p(new int(10));
  return p;
}

unique<int> p = func();         //It's ok.
{% endhighlight %} 

包含 unique\_ptr 的类对象的**默认复制构造函数**和**复制赋值函数被禁止；**

因为默认的复制构造函数会简单的复制 unique\_ptr 成员，而 unique\_ptr 是不允许被复制的。因此，只要是包含 unique\_ptr 的类都禁止
默认的复制构造函数。像上述例子中，Base中的上述提到的两个复制控制函数是被禁止掉的：

{% highlight c++ %}
void func(Base b) { return; }
func(*base_p);          //Error, Call the implicitly-deleted copy constructor of Baes;
Base b = *base_p        //Error, Call the implicitly-deleted copy constructor of Baes;
{% endhighlight %} 

但是，其并非禁止复制行为，如果用户自己定义的复制构造函数后，是可以进行复制构造的（只要不涉及到`unique_ptr`的直接复制）如果代码改为：

{% highlight c++ %}
class Base {
public:
Base(unique_ptr<int> int_p) {
    p = *(int_p.get());
    this->int_p = std::move(int_p);
}
Base(const Base& b) {
    printf("Copy Constructor\n");
    p = b.p;
}
private:
unique_ptr<int> int_p;
int p;
};

void func(Base b) { return; }
func(*base_p);          //It's OK
{% endhighlight %} 


### unique_ptr的好处及用途

**unique\_ptr for class member:**

如果类的某指针成员的所有权（ownership）只属于该类对象，那么应该将该成员定义为 unique\_ptr 。因为其可以保证
该指针所指的对象在类对象被销毁的时候也同时被销毁；此处优点有：

* 程序员不需要在析构函数显式的调用`delete`来进行内存管理；
* 禁止了默认拷贝构造函数和赋值构造函数.(或者督促程序员定义一个合适且安全的copy-constructor + operator=)


**unique\_ptr for local variable within functions:**

先看下列的这一个函数：

{% highlight c++ %}
void methodA() {
   int* buf = new int[256];

   int result = fillBuf(buf)) 
   if(result == -1) {
      return; 
   }
   printf("Result: %d", result);

   delete[] buf;
}
{% endhighlight %} 

上述的函数写法可能会有以下安全问题：

* return的时候忘记了delete `buf`,这会导致内存泄露；
* 加入在函数`fillBuf`中产生了异常，那么`buf`还是没有被delete并且会导致内存泄露；

如果将上述代码中的指针换成 unique_ptr，那么内存管理方面便是不需要用户过多的考虑了：

{% highlight c++ %}
void methodA() {
   unique_ptr<int> buf(new int[256]);

   int result = fillBuf(buf)) 
   if(result == -1) {
      return;
   }
   printf("Result: %d", result);
}
{% endhighlight %} 

* * * 

## shared_ptr

 shared_ptr 的作用有如同指针，但会记录有多少个`shared_ptrs`共同指向一个对象。这便是所谓的引用计数（reference counting）。一旦最后一个这样的指针被销毁，也就是一旦某个对象的引用计数变为0，这个对象会被自动删除。这在非环形数据结构中防止资源泄露很有帮助。

与`unique_ptr`不同， shared_ptr 是允许复制的。而且，上述提到的引用计数，是基于原子函数的的实现的，因此其是线程安全的.
当你在多线程编程时，如果不了解哪个线程会最迟结束（对象何时不在被需要），这时候最好的做法是每个线程单独赋予一个 shared_ptr .
但是值得注意的是，在这里所提及的`线程安全`,并不是指 shared_ptr 本身的线程安全的。如果两个线程同时访问一个 shared_ptr 对象的话，
那并不是线程安全的。其线程安全只是说每个线程有各自的 shared_ptr 对象的时候，引用计数的增减是线程安全的。

 shared_ptr 的问题在于假如存在循环依赖时：在一个双向链表中，假如A有一个指针指向B，B也有一个指针指向A，并且这些指针都是 shared_ptr 的话，则他们的reference count永远不会为0，因此会导致内存泄露；

下列是 shared_ptr 的一些使用介绍:

{% highlight c++ %}
struct MyClass {
   MyClass(const char* s);
   void methodA();
};
void someMethod(MyClass* m);

auto ptr = make_shared<MyClass>("obj1");

ptr->methodA();

someMethod(ptr.get());

shared_ptr<MyClass> anotherPtr = ptr; // now anotherPtr + ptr are both pointing to the "obj1" object

ptr.reset(new MyClass("obj2"); // now ptr switches to pointing to "obj2", but the "obj1" 
                               // object is not deleted as anotherPtr is still holding it

anotherPtr.reset(); // now no shared_ptr object is referencing the "obj1" MyClass*, so it is deleted

// "obj2" will be automically deleted when ptr goes out of scope
{% endhighlight %} 


* * * 

## weak_ptr

 weak_ptr 严格来说并不是一个smart pointer, 因为其不能使用`operator*`和`operator ->`,他对其指向的对象没有所有权，
也不负责维护对象的存在与消灭。它唯一的特性就是不会引起引用计数的增加，并且检查被管理目标物件是否存在，并且提供一个
接口产生一个临时的 shared_ptr 来使用；

 weak_ptr 一般是从 shared_ptr 复制产生，但是不会产生reference count.因此其主要用于打破上述所说 shared_ptr 的循环依赖。看下述代码：

{% highlight c++ %}
class Node;
typedef shared_ptr<Node> s_p;

class Node {

public:
    Node() {};
    Node(s_p next, s_p pre, int val)
        :next(next), pre(pre), val(val){}
    ~Node() { printf("Deconsturctor of value:%d\n", val); } 

    shared_ptr<Node> next;
    // weak_ptr<Node> pre;
    weak_ptr<Node> pre;
    int val;
};

shared_ptr<Node> build_list(const vector<int>& v) {
    shared_ptr<Node> head(NULL);
    shared_ptr<Node> cur(NULL);
    shared_ptr<Node> prev(NULL);
    for(auto i = 0; i < v.size(); i++) {
        if(head == NULL) {
            head = shared_ptr<Node>(new Node(NULL, NULL, v[i]));
            cur = head;
            prev = head;
        } else {
            cur->next = shared_ptr<Node>(new Node(NULL, prev, v[i]));
            cur = cur->next;
            prev = cur;
        }
    }
    return head;
}

int main() {
    auto head = build_list(vector<int>{1,2,3,4});
    auto itr = head;
    while(itr != NULL) {
        printf("%d->", itr->val);
        itr = itr->next;
    }
    printf("\n");
}
{% endhighlight %} 

上述代码是构建一个双向链表的代码，可以看到，上述的Node类的`pre`指针使用了 weak_ptr 来解除循环依赖。因此，上述代码的输出为：

{% highlight c++ %}
1->2->3->4->
Deconsturctor of value:1
Deconsturctor of value:2
Deconsturctor of value:3
Deconsturctor of value:4
{% endhighlight %} 

如果将`pre`的指针改为 shared_ptr ,程序的输出结果将没有了`Deconsturctor`的输出。因为到了`main`函数结束后，
`head`指针对象要被销毁，而此时`head`所指的对象引用技术减一。但是由于
`head`的后面的节点的`pre`指针依然指向他，导致`head`所指的对象因为引用计数没有到0而没有被销毁。同理于链表中的其他节点。


## 何时使用何种指针

这部分无力翻译了，更多的是一个个人的见解和个人的喜好，具体可以参考上述的第一个Reference
以及stackoverflow下的这个回答[Using smart pointers for class members](http://stackoverflow.com/questions/15648844/using-smart-pointers-for-class-members)：
