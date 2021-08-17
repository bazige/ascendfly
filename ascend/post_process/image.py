# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np

from ..common.const import *
from .color import color_val

def trans_numpy(img):
    """Trans AscendArray img to np.ndrray with rgb/bgr format.

    Args:
        img (AscendArray): The image to be transimited

    Returns:
        ndarray: The image with rgb/bgr format.
    """
    if isinstance(img, np.ndarray):
        return img

    # AscendArray image
    if img.format in yuv420:
        convert = cv2.COLOR_YUV2RGB_NV21 if img.format == PIXEL_FORMAT_YUV_SEMIPLANAR_420 else cv2.COLOR_YUV2RGB_NV12
        img = cv2.cvtColor(img.to_np, convert)
    else:
        img = img.to_np
    return img

def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (AscendArray or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    img = trans_numpy(img)
    cv2.imshow(win_name, img)

    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (AscendArray or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    img = trans_numpy(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (AscendArray, ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str, optional): Color of bbox lines. It also accept value of `Color`/str/tuple. 
            Defaults to 'green'.
        text_color (str, optional): Color of texts. It also accept value of `Color`/str/tuple. 
            Defaults to 'green'.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = trans_numpy(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = np.ascontiguousarray(img).astype(np.uint8)  # 可能传入的不是uint8
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img


def imshow_bboxes_colors(img,
                   bboxes,
                   class_names,
                   colors,
                   score_thr=0,
                   thickness=1,
                   font_scale=0.5,
                   show=True,
                   win_name='',
                   wait_time=0,
                   out_file=None):
    """Draw bboxes and class labels (with scores) on an image using various color.

    Args:
        img (AscendArray, ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores, label), shaped (n, 5) or (n, 6).
            shape like [x1, y1, x2, y2, scores, classid] or [x1, y1, x2, y2, classid]

        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 6
    assert len(colors) == len(class_names)
    img = trans_numpy(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 6
        scores = bboxes[:, 4]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]

    img = np.ascontiguousarray(img).astype(np.uint8)  # 可能传入的不是uint8

    for bbox, label in zip(bboxes[:,:5], bboxes[:,-1].astype('int32')):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, colors[label], thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, colors[label])

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img