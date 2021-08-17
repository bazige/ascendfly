[TOC]

# 1. Overview

## 1.1 Background

This project uses Python Ascend Computing Language (pyACL) APIs to implement the Ascendfly inference framework and encapsulates a series of easy-to-use Python APIs to simplify the PyACL development process and accelerate algorithm porting and deployment. The following describes the functions and dependencies of Ascendfly APIs and explains how to install and use them.

## 1.2 Functions

The project provides the following functions:

1. Encapsulate context and memory resource classes to simplify resource scheduling and allocation.
2. Encapsulate the AscendArray class, which is similar to using numpy.ndray to manage images and tensors on the device, to implement data consistency. AscendArray automatically manages data memory.
3. Encapsulate the VideoCapture and VideoWriter classes to obtain real-time H.264 (MAIN LEVEL without B frame) RTSP/RTMP streams and use Ascend 310 AI Processors to decode hardware or encode images into H.264/H.265 streams frame by frame.
4. Encapsulate the Image class to implement image processing functions such as image decoding, scaling, cropping, and padding.
5. Encapsulate the Model class to implement model inference.
6. Encapsulate the Profiling class to facilitate model performance tuning.
7. Provide other functions such as single-operator calling and post-processing.

## 1.3 Program Architecture

The system-level Ascendfly project includes the following modules:

1. Resource:

   The Resource module consists of mem, context, and thread/multi-process. mem is a memory object used to allocate and deallocate memory resources. context is used to allocate and deallocate context resources. multi-process is used for parallel acceleration.

2. Data:

   The Data module consists of AscendArray and Tensor. AscendArray implements AscendArray objects similar to numpy.ndarray. **It unifies the entire framework image and tensor data** and provides to_numpy, to_ascend, and clone methods that are different from ndarray to replicate numpy.ndarray data and objects. Tensor mainly implements the imgs2tensor and tensor2imgs functions to convert 3-dimensional images into 4-dimensional tensors for batch inference or tensor heatmap display.

3. Model:
   The Model module encapsulates AscendModel for model inference. After AscendModel is instantiated, model.tensor is used to obtain the input and output tensor names, feed_data is used to feed data to the model, run is used to implement inference, and get_tensor_by_name is used to obtain the output tensor data.

4. Image:

   The Image module involves image pre-processing (dynamic AIPP), image color space conversion (colorspace), image geometric transformation (geome), and image display (misc).

5. Video:
   The Video module encapsulates the VideoCapture class for H.264/H.265 RTSP/RTMP video decoding. The implementation of the Video module is basically the same as that of OpenCV VideoCapture: pyav pulls and decompresses streams, and vdec (encapsulating the DVPP decoding function) decodes the data. The Video module also encapsulates the VideoWriter class, which encodes single-frame YUV images into H.264/H.265 real-time video streams. The implementation of the Video module is slightly different from that of the OpenCV VideoWriter class: The single-frame image data of AscendArray is encoded by using venc (encapsulating the DVPP encoding function), and the video streams are saved to the local host.

6. Post_process:
   The Post_process module uses the bbox_overlaps function to calculate the Intersection over Union (IOU) of the bounding box (bbox), uses the nms function to calculate the NonMaximumSuppression, and uses the imshow and imshow_det_bboxes commands to display the detected target bbox or write the target bbox and confidence on the image and save the image.

5. Ops:
   The Ops module calls operators in the Blas library and operators such as Argmax, Cast, Transpose, and FFT.

7. Profiling
   The Profiling module encapsulates the Profiling class to simplify operator and model performance tuning, intuitively display the execution time of each operator, and sort the operators.

The following figure demonstrates the system design.

![Description of the input image]()

## 1.4 Design Process

![Description of the input image]()



# 2 Environment Dependencies and Installation Guide

## 2.1 Environment Dependencies

Ascendfly depends on **pyACL(CANN 21.0.1 or later)**, [**pyav**](https://github.com/PyAV-Org/PyAV), and [**PIL**](). The following describes how to install the dependent software.

Table 2-1 Environment requirements

| Environment Requirement| Description|
| -------- | ----------------------------------------------------------- |
| Hardware environment| Atlas 300 (model 3000 or 3010) or Atlas 800 (model 3000 or 3010)|
| OS| CentOS 7.6/Ubuntu 18.04|

Table 2-2 Environment-dependent software and software versions

| Software Name| Software Version|
| -------- | ------------------------------------- |
| pyACL  | (If CANN 21.0.1 or later is installed, pyACL is automatically installed.)|
| NumPy| ≥ 1.14 |
| pyav   | ≥ 8.0.2|
| PIL    | ≥ 8.0.2|
| objgraph  | ≥ 3.5.0 |
| prettytable | ≥ 2.1.0|
| (Optional) OpenCV| ≥ 3.4.2|



## 2.2 Installing CANN

As Python AscendCL programming APIs, pyACL opens functions such as context creation, memory allocation, model, and operator. The Ascendfly inference framework depends on the APIs provided by pyACL. For details about the environment installation method, see *[CANN Software Installation Guide](https://support.huaweicloud.com/instg-cli-cann/atlascli_03_0001.html)*. After installing CANN, [configure environment variables](https://support.huaweicloud.com/asdevg-python-cann/atlaspython_01_0006.html).

## 2.2 Installing Ascendfly
Ascendfly automatically installs related dependencies. You can run the following command to directly install them:

```shell
pip install ascendfly
```

## 2.3 (Optional) Installing OpenCV

If the ARM platform is used, install Python 3.7.5 before compiling and installing OpenCV-Python.

- **Step 1** Download OpenCV-Python.

   https://pypi.org/project/opencv-python/4.4.0.46/#files
   
- **Step 2** Decompress OpenCV-Python.

   tar -zxvf opencv-python-4.4.0.46.tar.gz && cd opencv-python-4.4.0.46
   
- **Step 3** Compile OpenCV-Python.

   python3.7.5 setup.py install

# 3 Usage Guide

## 3.1 Restrictions

This section describes the Ascendfly restrictions.

Table 3-1 Restrictions

| Item| Restriction |
| -------- | ------------------------------------------------------------ |
| pyACL    | For details, see [Application Development Guide (Python)](https://support.huaweicloud.com/intl/en-us/asdevg-python-cann/atlaspython_01_0001.html).|
| context| A context is created based on the device ID. Each context corresponds to a unique device ID.|
| VideoCapture | Currently, only the Real-Time Streaming Protocol (RTSP) is supported to pull H.264 video streams (excluding B frames).|



## 3.2 Preparations

Before running a model, convert the model and modify its configuration file. The storage locations of models and configuration files can be customized.

### 3.2.1 Preparing a Model

- **Step 1:** Download a model.

  Obtain the original network model, weight file, and **aipp_cfg** file (for details about how to download related models, visit the **tools/modelzoo** link), and save them to any directory of a common user in the development environment, for example, **/home/model/**.


- **Step 2:** Convert the model.

  Configure the AIPP by following the instructions in the [ATC Tool Instructions](https://support.huaweicloud.com/intl/en-us/tg-Inference-cann/atlasatc_16_0007.html) and convert the Caffe, TensorFlow, or ONNX model into an OM model of the Ascend platform. For details, see the **tools/convert** script.

### 3.2.2 Using Ascendfly APIs for Development

You can use Ascendfly APIs to develop the inference process based on the sample in the demo and *[API User Guide](#4.1)*.

## 3.3 Demo Running

Go to the **demo** directory, modify the configurations such as **device id** and **video_stream_path** of the demo to be run, and run the demo.

```shell
python3.7.5 yolov3_caffe_demo.py
```

   

# 4 Appendix
## 4.1 Ascendfly APIs
For details, see the **index.html** file in the **doc** directory of [Source Code](https://gitee.com/ascend-fae/ascendfly).

