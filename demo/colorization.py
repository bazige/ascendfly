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
import sys
sys.path.append("..")
import ascend
import numpy as np
import cv2
import time


class Colorization():
    def __init__(self, context, model_path):
        # initial model
        self.model = ascend.AscendModel(context, model_path)


    def pre_process(self, image):
        net_shape = self.model.tensor['data_l'].shape[2:4]
        image = cv2.resize(image, net_shape)
        image_norm = (1.0 * image / 255).astype(np.float32)
        lab = cv2.cvtColor(image_norm, cv2.COLOR_BGR2LAB)

        # pull out L channel and subtract 50 for mean-centering
        channels = cv2.split(lab)
        resize_L = channels[0] - 50
        return ascend.AscendArray.clone(resize_L)

    def detect(self, src_img):
        """ do model inference and detection-layer.
            @note
            the output feature-map should map with anchors.
        Args:
            src_img (AscendArray): input yuv padding image.

        Returns:
            detect result (ndarray).
        """
        # feed data and run inference
        self.model.feed_data({'data_l':src_img})
        t = time.time()
        self.model.run()
        print(time.time() - t)

        # get feature map of three output layer
        pred = self.model.get_tensor_by_name('class8_ab:0:class8_ab').to_np
        return pred


    def post_process(self, pred, image):
        h, w = image.shape[:2]
        mat_a, mat_b = pred[0][0, :], pred[0][1, :]

        image = (1.0 * image / 255).astype(np.float32)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        channels = cv2.split(lab)
        mat_a_up = cv2.resize(mat_a, (w, h))
        mat_b_up = cv2.resize(mat_b, (w, h))

        # merge image channel
        new_image = cv2.merge((channels[0], mat_a_up, mat_b_up))

        # convert back to rgb
        dst_img = cv2.cvtColor(new_image, cv2.COLOR_LAB2BGR)
        return dst_img
        

    def __del__(self):
        del self.model


def yolov5_detection():
    device_id = 0
    model_path = "../tools/model/colorization.om"
    video_stream_path = '/home/zhengguojian/ascendfly/tests/video/colorization.264'

    ctx = ascend.Context({device_id}).context_dict[device_id]
    # Img = ascend.Image(ctx)
    yolo = Colorization(ctx, model_path)
    
    # image = cv2.imread('../tests/image/zhuobielin.jpg')
    cap = ascend.VideoCapture(ctx, video_stream_path)
    while cap.is_open():
        yuv_img, _ = cap.read()
        if yuv_img:
            image = cv2.cvtColor(yuv_img.to_np, cv2.COLOR_YUV2RGB_NV21)
            data = yolo.pre_process(image)
            
            pred = yolo.detect(data)
            t1 = time.time()
            dst_img = yolo.post_process(pred, image)
            print('post:', time.time() - t1)
            cv2.imshow('colorization', dst_img)
            cv2.waitKey(5)


if __name__ == '__main__':
    yolov5_detection()
    