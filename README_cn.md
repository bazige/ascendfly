[TOC]

# 1 简介

## 1.1 背景

该项目通过Ascend Compute Language Python(pyACL) API实现Ascendfly推理框架，封装了一系列易用的python接口，目的是简化用户使用pyACL开发流程，加速算法迁移部署。以下对ascendfly相关接口功能、依赖安装和使用进行简要说明。

## 1.2 主要功能

本软件提供以下功能：

1. 封装了Context、Memory资源类，简化资源调度与分配。
2. 封装了AscendArray类，类似numpy.ndrray在device上进行图片和tensor管理，实现数据统一性。AscendArray自动管理数据内存，无需用户操作。
3. 封装了VideoCapture和VideoWriter类，获取实时H264（MAIN LEVEL without B frame）协议的RTSP/RTMP码流，并通过Ascend310芯片硬解码，或逐帧把图片编码为264/265码流。
4. 封装了Image类，实现图片解码、缩放、剪切、padding等图像处理功能。
5. 封装了Model类执行模型推理功能。
6. 封装了Profiling类，方便进行模型性能调优。
7. 其它如单算子调用，后处理等功能。

## 1.3 程序架构

Ascendfly系统级封装主要包括以下模块（module）。

1. 资源管理（resource）:

   resource包括内存模块（mem）、context和线程/进程资源（thread/multi-process）。mem主要是Memory对象，进行内存申请和释放；context主要实现Context资源申请和释放等；multiprocess主要做并行加速。

2. 数据模块（data）：

   主要包括ascendarray和tensor，ascendarray实现类似numpy.ndarray的AscendArray对象，**完成整个框架图像、tensor数据的统一性**，并具备不同于ndarray的to_numpy、to_ascend、clone方法实现numpy.ndarray数据和对象的复制。tensor主要实现imgs2tensor和tensor2imgs两个函数的功能，完成3维图片和4维tensor的转换，用于组batch进行推理或tensor heatmap显示。

3. 模型（model）：
   model模块封装了AscendModel，用来进行模型推理。模型实例化后，通过model.tensor获取输入输出tensor name，通过feed_data方法给模型数据，通过run方法实现推理，通过get_tensor_by_name获取输出tensor数据。

4. 图像模块（image）：

   涉及到图像预处理（动态aipp），图像色域空间转换（colorspace），图像几何变换（geome）和图像显示（misc）。

5. 视频模块（video)：
   video模块封装了VideoCapture类用于H264/H265 RTSP/RTMP视频解码，使用方式基本与opencv VideoCapture基本一致，实现方式是通过pyav进行拉流解包，vdec（封装了dvpp解码功能）进行解码。video模块还封装了VideoWriter类，用于把单帧yuv图像编码成H264/265的实时视频流，使用方式与opencv VideoWriter有轻微差别，实现方式是把AscendArray的单帧图像数据通过venc（封装了dvpp编码功能）进行编码，保存视频流到本地。

6. 后处理（post_process)：
   后处理部分实现了bbox_overlaps函数用于计算bbox的iou，nms用于计算NonMaximumSuppression，imshow、imshow_det_bboxes等显示检测的目标框或把目标框和confidence写在图片上并保存下来。

5. 算子模块（ops）：
   算子部分主要实现blas库算子调用和Argmax、Cast、Transpose、FFT等算子调用。

7. 性能调优（profiling)：
   模块封装了Profiling类可以更简单的实现算子、模型性能调优，可以直观地显示各算子执行耗时并加以排序。

整体系统设计如下图所示：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0222/151913_6ad4c066_8307159.jpeg "system.jpg")

## 1.4 设计流程

![输入图片说明](https://images.gitee.com/uploads/images/2021/0222/151932_3604f1a3_8307159.jpeg "thread.jpg")



# 2 环境依赖及安装指导

## 2.1 环境依赖

ascendfly需要依赖**pyACL（CANN 21.0.1及以上）**、[**pyav**](https://github.com/PyAV-Org/PyAV)和[**PIL**]()。以下简要介绍相关依赖软件安装过程。

表2-1 环境要求

| 环境要求 | 说明                                                        |
| -------- | ----------------------------------------------------------- |
| 硬件环境 | Atlas 300（型号3000 或 3010）/Atlas 800（型号3000 或 3010） |
| 操作系统 | CentOS 7.6/Ubuntu18.04                                      |

表2-2 环境依赖软件及软件版本

| 软件名称 | 软件版本                              |
| -------- | ------------------------------------- |
| pyACL  | （安装CANN 21.0.1 及以上，会自带安装pyACL ） |
| numpy  | >=1.14 |
| pyav   | >=8.0.2 |
| PIL    | >=8.2.0 |
| objgraph  | >=3.5.0 |
| prettytable | >=2.1.0 |
| opencv （可选）  | >=3.4.2 |



## 2.2 CANN安装

pyACL作为ACL python API编程接口，开放context创建、内存申请、模型和算子等功能，ascendfly推理框架依赖pyACL提供的API。具体环境安装方法参考[《CANN 软件安装指南》](https://support.huaweicloud.com/instg-cli-cann/atlascli_03_0001.html), 安装CANN后，进行[环境变量配置](https://support.huaweicloud.com/asdevg-python-cann/atlaspython_01_0006.html)。

## 2.2 Ascendfly安装
ascendfly会自动安装相关依赖，无需另外操作，通过以下命令直接安装

```shell
pip install ascendfly
```

## 2.3 opencv安装过程（可选）

如果是ARM平台，编译安装opencv-python前需要先安装python3.7.5

- **步骤 1** 下载opencv-python

   https://pypi.org/project/opencv-python/4.4.0.46/#files
   
- **步骤 2**  解压opencv-python

   tar -zxvf opencv-python-4.4.0.46.tar.gz && cd opencv-python-4.4.0.46
   
- **步骤 3**  编译opencv-python

   python3.7.5 setup.py install

# 3 使用指导

## 3.1 使用约束

本章节介绍Ascendfly限制约束。

表3-1 使用约束

| 名称     | 规格约束                                                     |
| -------- | ------------------------------------------------------------ |
| pyACL    | 请参考[《应用开发指南(Python)》](https://support.huaweicloud.com/asdevg-python-cann/atlaspython_01_0001.html) |
| context  | context通过device id创建，每个context对应唯一device id.                        |
| VideoCapture | 目前只支持rtsp协议，拉取H264（去除B帧）的视频流              |



## 3.2 使用前准备

运行前，需要先进行模型的转换和配置文件修改。模型和配置文件的存放位置可自定义。

### 3.2.1 模型准备

- **步骤 1 :** 模型下载

  首先，获取所用到的原始网络模型、权重文件和aipp_cfg文件（相关模型下载可参考tools/modelzoo链接），并将其存放到开发环境普通用户下的任意目录，例如：/home/model/


- **步骤 2:**  模型转换

  请参考[ATC工具参数说明](https://support.huaweicloud.com/tg-Inference-cann/atlasatc_16_0007.html)进行aipp配置，以及把caffe、TensorFlow或onnx模型转换为ascend平台om模型（可参考tools/convert脚本）。

### 3.2.2 利用ascendfly API进行开发

可参考demo中样例和[API使用手册](#4.1)，利用ascendfly API进行推理流程开发。

## 3.3 demo运行

进入demo目录下，对要运行demo的device id和video_stream_path等配置进行修改，运行测试demo

```shell
python3.7.5 yolov3_caffe_demo.py
```

   

# 4 附录
## 4.1 Ascendfly API
请参考[源码](https://gitee.com/ascend-fae/ascendfly) doc目录index.html文件。
