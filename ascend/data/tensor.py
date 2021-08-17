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

from ..common.const import *
from ..resource.mem import memcpy_d2d
from ..data.ascendarray import AscendArray
from ..ops.op import Permute


def _imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)

    # make a copy
    img = np.multiply(img, std)

    # inplace
    img = np.add(img, mean)
    if to_bgr:
        img = img[:, :, ::-1]
    return img


def imgs2tensor(imgs, tensor_fmt='NCHW', tensor_ptr=None):
    """Convert 3-channel images to tensor

    Args:
        imgs (list[AscendArray]): A list that contains multiple images,
            shape (h, w, c), support RGB/BGR, YUV444
        tensor_fmt (str, optional): Data format of output tensor. Defaults to 'NCHW'.
        tensor_ptr (int, optional): Data pointer of output tensor. If it is None, 
            we will create an AscendArray and bind the array's data pointer to it. 
            Defaults to None.

    Returns:
        AscendArray: Tensor that contains multiple images, shape (N, C, H, W) 
            or shape (N, H, W, C)

    Typical usage example:
    ```python
    imgs = [ascend_array1, ascend_array2]
    data = ascend.imgs2tensor(imgs, tensor_fmt='NHWC')
    ```
    """
    if not isinstance(imgs, list):
        raise TypeError(f"Input imgs expects a list, but got {type(imgs)}.")

    if len(imgs) <= 0:
        raise ValueError(f"Input imgs is a null list.")

    # get first image's shape and format
    format = imgs[0].format
    _shape = imgs[0].shape
    if format in yuv420:
        shape = _shape + (1,)
    else:
        shape = _shape

    # generate output tensor shape
    if tensor_fmt == 'NCHW':
        tensor_shape = (len(imgs),) + shape[-1:] + shape[:-1]
    elif tensor_fmt == 'NHWC':
        tensor_shape = (len(imgs),) + shape
    else:
        raise ValueError(
            f"Tensor format only accept 'NCHW' or 'NHWC', but got {tensor_fmt}.")

    if not tensor_ptr:
        tensor = AscendArray(
            tensor_shape, dtype=imgs[0].dtype, format=tensor_fmt)
        _ptr = tensor.ascend_data
    else:
        assert isinstance(tensor_ptr, int), \
            f"Input tensor_ptr expects an int, but got {type(tensor_ptr)}."
        _ptr = tensor_ptr

    nbytes = 0
    for i, img in enumerate(imgs):
        assert _shape == img.shape, f"imgs[{i}]'s shape {img.shape} is not same to others."
        assert format == img.format, f"imgs[{i}]'s format {img.shape} is not same to others."

        if tensor_fmt == 'NCHW':
            # swap channel using transform operator
            '''
            to do transformer
            '''
            pass

        nbytes = nbytes + img.nbytes
        memcpy_d2d(_ptr + nbytes, img.ascend_data, img.nbytes)

    return tensor if not tensor_ptr else None


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to a 3-channel images

    Args:
        tensor (AscendArray): Tensor that contains multiple images, shape (N, C, H, W) or shape (N, H, W, C)
        mean (tuple[float], optional): The mean value of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): The standard deviation of images. Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB format in the first place.
            If so, convert it back to BGR. Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.

    Typical usage example:
    ```python
    imgs = ascend.tensor2imgs(tensors)
    ```
    """
    if not isinstance(tensor, AscendArray):
        raise TypeError(
            f"Input tensor expects an AscendArray, but got {type(tensor)}.")

    if tensor.ndim != 4:
        raise ValueError(
            f"Input tensor expects a 4-dim, but got {tensor.ndim}.")

    if tensor.format not in ["NCHW", "NHWC"]:
        raise ValueError(
            f"Input tensor's format only support 'NCHW' or 'NHWC', but given {tensor.format}.")

    assert len(mean) == 3, \
        f"Input mean of images expects a 3-elements tuple, but got {len(mean)}."
    assert len(std) == 3, \
        f"Input std of images expects a 3-elements tuple, but got {len(std)}."

    batch_size = tensor.shape[0]
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    if tensor.format == "NCHW":
        try:
            tensor = Permute(tensor, axes=(0, 2, 3, 1))
        except:
            tensor = tensor.to_np.transpose(0, 2, 3, 1)
    else:
        tensor = tensor.to_np

    imgs = []
    for img_id in range(batch_size):
        img = tensor[img_id, ...]
        img = _imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs
