#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

# error code
ACL_SUCCESS = 0

# rule for mem
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# NUMPY data type
# ptr_to_numpy
NPY_BOOL = 0
NPY_BYTE = 1
NPY_UBYTE = 2
NPY_SHORT = 3
NPY_USHORT = 4
NPY_INT = 5
NPY_UINT = 6
NPY_LONG = 7
NPY_ULONG = 8
NPY_LONGLONG = 9
NPY_ULONGLONG = 10
NPY_FLOAT32 = 11
NPY_DOUBLE = 12
NPY_FLOAT16 = 23

# data format
ACL_FORMAT_UNDEFINED = -1
ACL_FORMAT_NCHW = 0
ACL_FORMAT_NHWC = 1
ACL_FORMAT_ND = 2
ACL_FORMAT_NC1HWC0 = 3
ACL_FORMAT_FRACTAL_Z = 4


# data type
# get_input_data_type
ACL_DT_UNDEFINED = -1
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3
ACL_UINT8 = 4
ACL_INT16 = 6
ACL_UINT16 = 7
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_DOUBLE = 11
ACL_BOOL = 12


ACL_DDR_MEM           = 0 # DDR内存，DDR上所有大页内存+普通内存 
ACL_HBM_MEM           = 1 # HBM内存，HBM上所有大页内存+普通内存 
ACL_DDR_MEM_HUGE      = 2 # #DDR大页内存 
ACL_DDR_MEM_NORMAL    = 3 # DDR普通内存
ACL_HBM_MEM_HUGE      = 4 # HBM大页内存 
ACL_HBM_MEM_NORMAL    = 5 # HBM普通内存 
ACL_DDR_MEM_P2P_HUGE  = 6 # DDR中用于Device间数据复制的 大页内存 
ACL_DDR_MEM_P2P_NORMAL= 7 # DDR中用于Device间数据复 制的普通内存 
ACL_HBM_MEM_P2P_HUGE  = 8 # HBM中用于Device间数据复制 的大页内存 
ACL_HBM_MEM_P2P_NORMAL= 9 # HBM中用于Device间数据复 制的普通内存

# pyav stream.codec_context profile
PROFILE_HEVC_MAIN = 1
PROFILE_H264_BASE = 66
PROFILE_H264_MAIN = 77
PROFILE_H264_HIGH = 100

# video encoding protocol
H265_MAIN_LEVEL = 0
H264_BASELINE_LEVEL = 1
H264_MAIN_LEVEL = 2
H264_HIGH_LEVEL = 3


# dvpp pixel format
PIXEL_FORMAT_YUV_400 = 0  # YUV400 8bit
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1  # YUV420SP NV12 8bit
PIXEL_FORMAT_YVU_SEMIPLANAR_420 = 2  # YUV420SP NV21 8bit
PIXEL_FORMAT_YUV_SEMIPLANAR_422 = 3  # YUV422SP NV12 8bit
PIXEL_FORMAT_YVU_SEMIPLANAR_422 = 4  # YUV422SP NV21 8bit
PIXEL_FORMAT_YUV_SEMIPLANAR_444 = 5  # YUV444SP NV12 8bit
PIXEL_FORMAT_YVU_SEMIPLANAR_444 = 6  # YUV444SP NV21 8bit
PIXEL_FORMAT_YUYV_PACKED_422 = 7  # YUV422P YUYV 8bit
PIXEL_FORMAT_UYVY_PACKED_422 = 8  # YUV422P UYVY 8bit
PIXEL_FORMAT_YVYU_PACKED_422 = 9  # YUV422P YVYU 8bit
PIXEL_FORMAT_VYUY_PACKED_422 = 10  # YUV422P VYUY 8bit
PIXEL_FORMAT_YUV_PACKED_444 = 11  # YUV444P 8bit
PIXEL_FORMAT_RGB_888 = 12  # RGB888
PIXEL_FORMAT_BGR_888 = 13  # BGR888
PIXEL_FORMAT_ARGB_8888 = 14  # ARGB8888
PIXEL_FORMAT_ABGR_8888 = 15  # ABGR8888
PIXEL_FORMAT_RGBA_8888 = 16  # RGBA8888
PIXEL_FORMAT_BGRA_8888 = 17  # BGRA8888
PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT = 18  # YUV420SP 10bit
PIXEL_FORMAT_YVU_SEMI_PLANNER_420_10BIT = 19  # YVU420sp 10bit
PIXEL_FORMAT_YVU_PLANAR_420 = 20  # YUV420P 8bit
PIXEL_FORMAT_YVU_PLANAR_422 = 21
PIXEL_FORMAT_YVU_PLANAR_444 = 22
PIXEL_FORMAT_RGB_444 = 23
PIXEL_FORMAT_BGR_444 = 24
PIXEL_FORMAT_ARGB_4444 = 25
PIXEL_FORMAT_ABGR_4444 = 26
PIXEL_FORMAT_RGBA_4444 = 27
PIXEL_FORMAT_BGRA_4444 = 28
PIXEL_FORMAT_RGB_555 = 29
PIXEL_FORMAT_BGR_555 = 30
PIXEL_FORMAT_RGB_565 = 31
PIXEL_FORMAT_BGR_565 = 32
PIXEL_FORMAT_ARGB_1555 = 33
PIXEL_FORMAT_ABGR_1555 = 34
PIXEL_FORMAT_RGBA_1555 = 35
PIXEL_FORMAT_BGRA_1555 = 36
PIXEL_FORMAT_ARGB_8565 = 37
PIXEL_FORMAT_ABGR_8565 = 38
PIXEL_FORMAT_RGBA_8565 = 39
PIXEL_FORMAT_BGRA_8565 = 40
PIXEL_FORMAT_RGB_BAYER_8BPP = 50
PIXEL_FORMAT_RGB_BAYER_10BPP = 51
PIXEL_FORMAT_RGB_BAYER_12BPP = 52
PIXEL_FORMAT_RGB_BAYER_14BPP = 53
PIXEL_FORMAT_RGB_BAYER_16BPP = 54
PIXEL_FORMAT_BGR_888_PLANAR = 70
PIXEL_FORMAT_HSV_888_PACKAGE = 71
PIXEL_FORMAT_HSV_888_PLANAR = 72
PIXEL_FORMAT_LAB_888_PACKAGE = 73
PIXEL_FORMAT_LAB_888_PLANAR = 74
PIXEL_FORMAT_S8C1 = 75
PIXEL_FORMAT_S8C2_PACKAGE = 76
PIXEL_FORMAT_S8C2_PLANAR = 77
PIXEL_FORMAT_S16C1 = 78
PIXEL_FORMAT_U8C1 = 79
PIXEL_FORMAT_U16C1 = 80
PIXEL_FORMAT_S32C1 = 81
PIXEL_FORMAT_U32C1 = 82
PIXEL_FORMAT_U64C1 = 83
PIXEL_FORMAT_S64C1 = 84
PIXEL_FORMAT_BUTT = 1003
PIXEL_FORMAT_UNKNOWN = 10000

DVPP = 1
DEVICE = 2
HOST = 3

#log
DEBUG = 0
INFO = 1
WARNING = 2
ERROR = 3

interp_codes = {
    'huawei': 0,
    'bilinear': 1,
    'nearest': 2,
    'bicubic': 3,
    'area': 4,
    'lanczos': 5
}

# images format
IMG_EXT_ENC = [
    '.png',
    '.PNG',
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG'
]

yuv420 = [
    PIXEL_FORMAT_YUV_SEMIPLANAR_420, 
    PIXEL_FORMAT_YVU_SEMIPLANAR_420
]

pix_fmt_map = {
    'yuv400'   : PIXEL_FORMAT_YUV_400,
    'yuv420p'  : PIXEL_FORMAT_YUV_SEMIPLANAR_420,
    'yuvj420p' : PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    'yuv422p'  : PIXEL_FORMAT_YUV_SEMIPLANAR_422,
    'yuvj422p' : PIXEL_FORMAT_YVU_SEMIPLANAR_422,
    'yuv444p'  : PIXEL_FORMAT_YUV_SEMIPLANAR_444,
    'yuvj444p' : PIXEL_FORMAT_YVU_SEMIPLANAR_444,

    'rgb'  : PIXEL_FORMAT_RGB_888, # RGB88
    'bgr'  : PIXEL_FORMAT_BGR_888, # BGR88
    'argb' : PIXEL_FORMAT_ARGB_8888, # ARG
    'abgr' : PIXEL_FORMAT_ABGR_8888, # ABG
    'rgba' : PIXEL_FORMAT_RGBA_8888, # RGB
    'bgra' : PIXEL_FORMAT_BGRA_8888  # BGR
}

en_type_map = {
    PROFILE_HEVC_MAIN : H265_MAIN_LEVEL,
    PROFILE_H264_BASE : H264_BASELINE_LEVEL,
    PROFILE_H264_MAIN : H264_MAIN_LEVEL,
    PROFILE_H264_HIGH : H264_HIGH_LEVEL
}


dtype_dict = {
    ACL_FLOAT  : np.dtype('float32'),
    ACL_FLOAT16: np.dtype('float16'),
    ACL_INT8   : np.dtype('int8')   ,
    ACL_INT32  : np.dtype('int32')  ,
    ACL_UINT8  : np.dtype('uint8')  ,
    ACL_INT16  : np.dtype('int16')  ,
    ACL_UINT16 : np.dtype('uint16') ,
    ACL_UINT32 : np.dtype('uint32') ,
    ACL_INT64  : np.dtype('int64')  ,
    ACL_UINT64 : np.dtype('uint64') ,
    ACL_DOUBLE : np.dtype('float64'),
    ACL_BOOL   : np.dtype('bool')   
}

tensor_fmt_map = {
    ACL_FORMAT_UNDEFINED: 'UNDEFINED',
    ACL_FORMAT_NCHW     : 'NCHW',
    ACL_FORMAT_NHWC     : 'NHWC',
    ACL_FORMAT_ND       : 'ND',
    ACL_FORMAT_NC1HWC0  : 'NC1HWC0',
    ACL_FORMAT_FRACTAL_Z: 'FRACTAL_Z'
}

tensor_fmt_dict = {
    'UNDEFINE'  : ACL_FORMAT_UNDEFINED,
    'NCHW'      : ACL_FORMAT_NCHW,
    'NHWC'      : ACL_FORMAT_NHWC,
    'ND'        : ACL_FORMAT_ND,
    'NC1HWC0'   : ACL_FORMAT_NC1HWC0,
    'FRACTAL_Z' : ACL_FORMAT_FRACTAL_Z
}

numpy_dict = {
    np.dtype('float32') : NPY_FLOAT32,
    np.dtype('float16') : NPY_FLOAT16,
    np.dtype('int8')    : NPY_BYTE,
    np.dtype('int32')   : NPY_INT,
    np.dtype('uint8')   : NPY_UBYTE,
    np.dtype('int16')   : NPY_SHORT,
    np.dtype('uint16')  : NPY_USHORT,
    np.dtype('uint32')  : NPY_UINT,
    np.dtype('int64')   : NPY_LONG,
    np.dtype('uint64')  : NPY_ULONG,
    np.dtype('float64') : NPY_DOUBLE,
    np.dtype('bool')    : NPY_BOOL
}

