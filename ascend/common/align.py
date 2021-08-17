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

from .const import *

GRAY_CHN = [
    PIXEL_FORMAT_YUV_400
]

SEMI_CHN = [
    PIXEL_FORMAT_YUV_SEMIPLANAR_420,
    PIXEL_FORMAT_YVU_SEMIPLANAR_420
]

TWO_CHN = [
    PIXEL_FORMAT_YUV_SEMIPLANAR_422,
    PIXEL_FORMAT_YVU_SEMIPLANAR_422,
    PIXEL_FORMAT_YUYV_PACKED_422,
    PIXEL_FORMAT_UYVY_PACKED_422,
    PIXEL_FORMAT_YVYU_PACKED_422,
    PIXEL_FORMAT_VYUY_PACKED_422
]

THR_CHN = [
    PIXEL_FORMAT_YUV_SEMIPLANAR_444,
    PIXEL_FORMAT_YVU_SEMIPLANAR_444,
    PIXEL_FORMAT_YUV_PACKED_444,
    PIXEL_FORMAT_RGB_888,
    PIXEL_FORMAT_BGR_888
]

FUR_CHN = [
    PIXEL_FORMAT_ARGB_8888,
    PIXEL_FORMAT_ABGR_8888,
    PIXEL_FORMAT_RGBA_8888,
    PIXEL_FORMAT_BGRA_8888
]


def aligned(value, func='vdec', aligned='w'):
    """Align specific size of 'w' or 'h' according to input func.

    Args:
        value (int): Input value to be aligned.
        func (str, optional): Alinged method for specified function. Defaults to 'vdec'.
        aligned (str, optional): Aligne `w` or `h`. Defaults to 'w'.
    
    Raises:
        ValueError: Input `func` not in ['jpegd', 'pngd', 'vdec', 'resize', 'crop', 'crop_and_paste', 'encode']
        ValueError: Input aligned not in ['w', 'h']

    Returns:
        [int]: A aligned value.
    """
    if func not in ['jpegd', 'pngd', 'vdec', 'resize', 'crop', 'crop_and_paste', 'encode']:
        raise ValueError(
            f"Aligned func only support:'jpegd','vdec','resize','crop','crop_and_paste'.")

    if aligned not in ['w', 'h']:
        raise ValueError(f"Aligned func only support w or h aligned.")

    if func == 'jpegd' or func == 'pngd':
        if aligned == 'w':
            alignment = 128
        else:
            alignment = 16
    else:
        if aligned == 'w':
            alignment = 16
        else:
            alignment = 2

    formal = ((value + alignment - 1) // alignment) * alignment
    return formal


def calc_size(width, height, iformat):
    """Caculating image nbytes according input width and height.

    Args:
        width (int): Aligned image width.
        height (int): Aligned image height.
        iformat (int): Image formats, see common/const.py for details.

    Returns:
        [int]: The caculated image size.
    """
    if not (isinstance(width, int) and width > 0):
        raise TypeError(
            f"input width expects an positive int, but got {type(width)}.")

    if not (isinstance(height, int) and height > 0):
        raise TypeError(
            f"input height expects an positive int, but got {type(height)}.")

    if not isinstance(iformat, int):
        raise TypeError(
            f"input iformat expects an int, but got {type(iformat)}.")

    if width <= 0 or height <= 0:
        width = 0
        height = 0

    if iformat in GRAY_CHN:
        size = width * height
    elif iformat in SEMI_CHN:
        size = width * height * 3 // 2
    elif iformat in TWO_CHN:
        size = width * height * 2
    elif iformat in THR_CHN:
        size = width * height * 3
    elif iformat in FUR_CHN:
        size = width * height * 4
    else:
        raise ValueError(f"Input data format:{iformat} is invalid.")

    return size
