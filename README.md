## 基于C++的深度学习框架: minnet



### 1. 框架配置

本框架基于C++20标准实现，除此之外没有其他外部依赖。编译项目需要所在机器安装有cmake。

在根目录运行：

```shell
mkdir build
&& cd build
&& cmake ..
```

完成项目构建，构建完成后可以在{根目录}/build/src目录下找到对应的静态链接库文件lib_minnet.a。如果运行机器安装有OpenCV您也可以在{根目录}/build/example目录下找到对应的演示程序，演示程序将基于mnist数据集进行训练并且输出测试结果。

### 2. 框架结构

//TODO()