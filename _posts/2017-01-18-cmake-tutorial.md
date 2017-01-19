---
layout: post
title: "CMake初入门杂谈"
modified:
categories: Linux
description:
tags: [CMake]
comments: true
mathjax: true
share:
---

说来惭愧，在Unix系统下工作多年，一直没有认真的学习过 CMake，仅仅限于能简单看懂的地步。近来由于尝试将开源 c++ 的 [Json11](https://github.com/dropbox/json11) 转换为 Python 拓展，并对比各种工具转换的区别（Boost.Python, SWIG, pybind...）。因为需要利用 CMake 来有效的管理和组织项目，因此便花了些时间，做了 CMake 的一些简单入门；

### CMake 入门

网上的教程多如牛毛，此处便不会再用自己蹩脚的语言来重新组织一下了。推荐几个相当不错的 Tutorial：

* [CMake-Tutorial](https://cmake.org/cmake-tutorial/)
* [CMake 入门实战](http://hahack.com/codes/cmake/)

认真阅读并实践上述 Tutorial 后，对理解 CMake 会有着不少的帮助。在 CMake 的使用过程中，我们常常会遇到很多 CMake 预先定义好的各式变量，本段落针对这些变量，做出一些例子和详述。

**下面段落更多是从文章[CMake Useful Variables](https://cmake.org/Wiki/CMake_Useful_Variables)挑出部分重点进行翻译和补充说明。**

#### `CMAKE_BINARY_DIR`

> if you are building in-source, this is the same as `CMAKE_SOURCE_DIR`, otherwise this is the top level directory of your build tree

所谓的 `building in-source`, 是指 build(运行命令`CMake`) 的路径和 `CMakeLists.txt`文件的路径一致。 而 top-level-directory-of-your-build-tree, 是执行指 build 的路径, eg：

```python
CMakeList.txt    build/ include/ ...
# 假如位于文件夹 build/ 内进行 build，则 `CMAKE_BINARY_DIR` 则是 build 文件夹的绝对路径
```

#### `CMAKE_CURRENT_BINARY_DIR`

> if you are building in-source, this is the same as `CMAKE_CURRENT_SOURCE_DIR`, otherwise this is the directory where the compiled or generated files from the current CMakeLists.txt will go to

比起上一个变量，这个变量多出了 CURRENT 的字眼。从上面的教程中可以知道，很多情况下，一个项目中会存在多个 CMakeLists.txt，（例如需要将功能代码编译成静态库时），主 CMakeLists.txt 通过利用CMake命令 `add_subdirectory` 来添加子目录并处理其中的 CMakeList.txt。因此，该变量是指当前被处理的 CMakeLists.txt (例如子目录下的) 编译/生成的文件所处的位置；

#### `CMAKE_CURRENT_SOURCE_DIR`

> this is the directory where the currently processed CMakeLists.txt is located in

当前处理的 CMakeList.txt 所处的位置；

#### `CMAKE_MODULE_PATH`

> tell CMake to search first in directories listed in `CMAKE_MODULE_PATH` when you use `FIND_PACKAGE()` or `INCLUDE()`

每次 CMake 中调用 `FIND_PACKAGE()` 或者 `INCLUDE()` 指令，都会先从 `CMAKE_MODULE_PATH` 变量中的地址先搜索。例如：

```cmake
# 利用set设置 `CMAKE_MODULE_PATH`， 在这里set的用法是list的用法，令
# `CMAKE_MODLUE_PATH`  = [旧的`CMAKE_MODULE_PATH`, ${CMAKE_SOURCE_DIR}/cmake/Modules/]
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# 假设 cmake/Modules/ 下有 FindHelloWorld.cmake 文件
FIND_PACKAGE(HelloWorld)
```

最后，关于 cmake 的 `find_package` 是如何工作的，有兴趣的可以阅读下官方的wiki [CMake:How To Find Libraries](https://cmake.org/Wiki/CMake:How_To_Find_Libraries)

#### `CMAKE_SOURCE_DIR`

> this is the directory which contains the top-level CMakeLists.txt, i.e. the top level source directory

包含最顶层 CMakeLists.txt 的目录

#### `RUNTIME_OUTPUT_DIRECTORY` / `LIBRARY_OUTPUT_DIRECTORY` / `ARCHIVE_OUTPUT_DIRECTORY`

> 值得注意的是，上述的三个变量是新的写法，对应旧的变量 `EXECUTEABLE_OUTPUT_PATH/LIBRARIY_OUTPUT_PATH`

通过修改上面三个变量的值，可以使得 CMAKE 将生成的库（静态库/动态库），可执行文件分别放在不同的位置，而不是
默认的值 `CMAKE_CURRENT_BINARY_DIR`，例如(对应 stackoverflow 上的一个[回答](http://stackoverflow.com/questions/6594796/how-do-i-make-cmake-output-into-a-bin-dir))：

```cmake
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# You can also specify the output directories on a per target basis:

set_target_properties( targets...
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
```

### 一个例子： 利用CMake管理 Boost.Python 程序

下面的代码，是一个简单利用 CMake 来管理 Boost.Python 程序编译过程的代码

```cmake
cmake_minimum_required(VERSION 3.0) 

project(hello_ext)
# 依赖对应的Python api 并 将对应的header路径include近来
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# 查找boost.python 库
find_package(Boost 1.59.0 COMPONENTS python)
if (Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_library(hello_ext SHARED test.cpp)
  # Do not follow the convention for the dynamic libraries filename, change
  # the libraries filename from 'libhello_ext.dylib' -> 'hello_ext.so'
  # 这样做的原因是因为：
  # 1.python中import的库的名字，必须和 `BOOST_PYTHON_MODULE()`中定义的模块名字一致；
  # 2.mac下生成的dylib后缀的动态库无法import到python中，需要改为 .so 后缀的。
  set_target_properties(hello_ext PROPERTIES PREFIX "" SUFFIX ".so")
  target_link_libraries(hello_ext ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

elseif(not Boost_FOUND)
  message(FATAL_ERROR "Unable to find correct Booost Python Components")
endif()
```

### Vim下写CMake的插件

[richq/vim-cmake-completion](https://github.com/richq/vim-cmake-completion)该插件通过提供 针对 CMake语言的 `omnifunc function`, 使得在 Vim 中可以通过 Ctrl-x Ctrl-o 来进行补全对应的命令以及命令内的参数；同时，对相应的命令按下K键，也可以提供对应命令的文档窗口：如图

![K键显示对应命令的文档说明](http://ok0mhspkg.bkt.clouddn.com/WX20170119-151606@2x.png)

