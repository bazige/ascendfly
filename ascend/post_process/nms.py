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

def nms(boxes, scores, nms_thresh=0.3):
    """Suppress overlapping detected bbox.

    Args:
        boxes (ndarray): Input bboxes with shape (n, 4)
        scores (ndarray): Input scores with shape (k, 4) of bboxes
        nms_thresh (float, optional): The threshold of iou (intersection over union). 
            Defaults to 0.3.

    Returns:
        keep (ndarray): The result of nms
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    num_det = boxes.shape[0]
    suppressed = np.zeros((num_det,), dtype=np.bool)

    keep = []
    for _i in range(num_det):
        i = order[_i]
        if suppressed[i]:
            continue
        keep.append(i)

        ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_det):
            j = order[_j]
            if suppressed[j]:
                continue

            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= nms_thresh:
                suppressed[j] = True

    return keep