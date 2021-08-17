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
import acl
import numpy as np
from ..common.const import *
from ..data.ascendarray import AscendArray
from ..ops.op import Permute


def show_img(imgs, window_names=None, wait_time_ms=0):
    """Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Notice, specified format must be greater than or equal to imgs number. When you using this 
        function, you'd better install opencv first.

    Args:
        imgs (AscendArray, ndarray or list): Input images.
        window_names (str, optional): If None, function will create different windows as '1', '2'.
        wait_time_ms (int, optional): Display wait time.

    Typical usage example:
    ```python
    ascend.show_img(image, wait_time_ms=15)
    ```
    """
    try:
        import cv2
    except ImportError:
        Log(ERROR, f"import cv2 error while using show_img, and you should install opencv first.")
        return

    if not isinstance(imgs, list):
        imgs = [imgs]

    if window_names is None:
        window_names = list(range(len(imgs)))
    else:
        if not isinstance(window_names, list):
            window_names = [window_names]
        assert len(imgs) == len(window_names), 'window names does not match images!'

    show_imgs = []
    for i, img in enumerate(imgs):
        show_imgs.append(img.to_np if isinstance(img, AscendArray) else img)
 
    for img, win_name in zip(show_imgs, window_names):
        if img is None:
            continue
        win_name = str(win_name)
        cv2.namedWindow(win_name, 0)
        cv2.imshow(win_name, img)

    cv2.waitKey(wait_time_ms)


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


def random_color(rgb=False, maximum=255):
    """Generate random color

    Args:
        rgb (bool): Whether to return RGB colors or BGR colors.
        maximum (int): Either 255 or 1

    Returns:
        ndarray: A vector of 3 numbers

    Typical usage example:
    ```python
    color = ascend.random_color()
    ```
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def show_bbox(image, bboxs_list, color=None, thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
              is_show=True, is_without_mask=False):
    """Visualize bbox in object detection by drawing rectangle. Notice, when you using this function, 
    you'd better install opencv first.

    Args:
        image (AscendArray): Input image to be drawed with lines and labels.
        bboxs_list (list): A list with [pts_xyxy, prob, id], the preditction
        color (tuple, optional): The bbox's color. Defaults to None.
        thickness (int, optional): The thickness of line. Defaults to 1.
        font_scale (float, optional): The font scale of bbox. Defaults to 0.3.
        wait_time_ms (int, optional): Image's views time interval. Defaults to 0.
        names ([type], optional): Figure's window name. Defaults to None.
        is_show (bool, optional): Show the image or not. Defaults to True.
        is_without_mask (bool, optional): Defaults to False.

    Returns:
        ndarray: A image with bbox and text

    Typical usage example:
    ```python
    ascend.show_bbox(image, bboxes, wait_time_ms=10)
    ```
    """
    try:
        import cv2
    except ImportError:
        Log(ERROR, f"import cv2 error while using show_bbox, and you should install opencv first.")
        return

    if not isinstance(image, AscendArray):
        raise TypeError(f"Input image expects an AscendArray, but got {type(image)}.")

    font = cv2.FONT_HERSHEY_SIMPLEX
    image_copy = image.to_np
    if image.format == PIXEL_FORMAT_YUV_SEMIPLANAR_420:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_YUV2RGB_NV21)
    elif image.format == PIXEL_FORMAT_YVU_SEMIPLANAR_420:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_YUV2RGB_NV12)

    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f},id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)
        if color is None:
            colors = random_color(rgb=True).astype(np.float64)
        else:
            colors = color

        if not is_without_mask:
            image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors,
                                       thickness)
        else:
            mask = np.zeros_like(image_copy, np.uint8)
            mask1 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, -1)
            mask = np.zeros_like(image_copy, np.uint8)
            mask2 = cv2.rectangle(mask, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), colors, thickness)
            mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
            image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(image_copy, txt, (bbox_f[0], bbox_f[1] - 2),
                        font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
    if is_show:
        show_img(image_copy, names, wait_time_ms)
    return image_copy


def show_tensor(tensor, resize_hw=None, is_show=True, wait_time_ms=0):
    """Show tensor as heatmap.

    Args:
        tensor (AscendArray): Input tensor
        resize_hw (tuple, optional): Wether to resize the input tensor to fit the window. Defaults to None.
        is_show (bool, optional): Show the tensor or save it. Defaults to True.
        wait_time_ms (int, optional): Display time. Defaults to 0.

    Returns:
        list[AscendArray]: Converted tensor heatmap.

    .. warning::
        When you using this function, you'd better install opencv first.

    Typical usage example:
    ```python
    ascend.show_tensor(tensor, resize_hw=(640, 480), wait_time_ms=10)
    ```
    """
    try:
        import cv2
    except ImportError:
        Log(ERROR, f"import cv2 error while using show_tensor, and you should install opencv first.")
        return

    def normalize_numpy(array):
        max_value = np.max(array)
        min_value = np.min(array)
        array = (array - min_value) / (max_value - min_value)
        return array

    if not isinstance(tensor, AscendArray):
        raise TypeError(f"Input tensor expects an AscendArray, but got {type(tensor)}.")
        
    if tensor.ndim != 4:
        raise ValueError('Dim of input tensor should be 4, please check your tensor dimension!')

    if tensor.format == 'NCHW':
        tensor = tensor
    else:
        '''
        to transpose NCHW, using ascend310 first. if it fails, use numpy.
        '''
        try:
            tensor = Permute(tensor, axes=(0, 2, 3, 1))
        except:
            tensor = tensor.to_np.transpose((0, 2, 3, 1))

    # resize the tensor with interpolize
    if resize_hw is not None:
        pass

    tensor = tensor.permute(1, 2, 0)

    channel = tensor.shape[2]
    tensor = tensor.to_np
  
    # do normalize
    sum_tensor = np.sum(tensor, axis=2)
    sum_tensor = normalize_numpy(sum_tensor) * 255
    sum_tensor = sum_tensor.astype(np.uint8)

    # show tensor as colormap
    sum_tensor = cv2.applyColorMap(np.uint8(sum_tensor), cv2.COLORMAP_JET)
    # mean_tensor = cv2.applyColorMap(np.uint8(mean_tensor), cv2.COLORMAP_JET)

    if is_show:
        show_img([sum_tensor], ['sum'], wait_time_ms=wait_time_ms)
    return [sum_tensor]

