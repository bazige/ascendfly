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
import os
sys.path.append("..")
import ascend
import numpy as np
import cv2

MODEL_WIDTH = 416
MODEL_HEIGHT = 416

def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT, 
                           MODEL_WIDTH, MODEL_HEIGHT], 
                           dtype = np.float32) 
    
    return ascend.AscendArray.clone(image_info)


def post_process(bbox_num, bbox, origin_img):
    """postprocess"""
    print("post process")
    box_num = bbox_num[0, 0]
    box_info = bbox.flatten()

    scalex = origin_img.shape[1] / MODEL_WIDTH
    scaley = origin_img.shape[0] / MODEL_HEIGHT
    if scalex > scaley:
        scaley =  scalex
    
    bboxes = []
    labels_id = []
    for n in range(int(box_num)):
        ids = int(box_info[5 * int(box_num) + n])
        labels_id.append(ids)
        score = box_info[4 * int(box_num)+n]
        top_left_x = box_info[0 * int(box_num)+n] * scaley
        top_left_y = box_info[1 * int(box_num)+n] * scaley
        bottom_right_x = box_info[2 * int(box_num) + n] * scaley
        bottom_right_y = box_info[3 * int(box_num) + n] * scaley
        # print("class % d, box % d % d % d % d, score % f" % (
        #     ids, top_left_x, top_left_y, 
        #     bottom_right_x, bottom_right_y, score))
        bboxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y, score])
        
    return np.array(bboxes), np.array(labels_id)



def yolov3_caffe():
    device_id = 0
    model_path = "../tools/model/yolov3_aipp.om"
    video_stream_path = '/home/zhengguojian/demo/tests/video/changsha_wuyi_road.264'
    print(video_stream_path)

    ctx = ascend.Context({device_id}).context_dict[device_id]
    Img = ascend.Image(ctx)
    cap = ascend.VideoCapture(ctx, video_stream_path)
    mdl = ascend.AscendModel(ctx, model_path)

    image_info = construct_image_info()
    colors = ascend.color_gen(len(ascend.coco_classes()))
    while cap.is_open():
        yuv_img, frame_id = cap.read()
        if yuv_img:
            yuv_resized = Img.imrescale(yuv_img, (416, 416))
            yuv_pad = Img.impad(yuv_resized, shape=(416, 416))
            origin_img = cv2.cvtColor(yuv_img.to_np, cv2.COLOR_YUV2RGB_NV21)
            origin_resize  = cv2.resize(origin_img, (640, 360))
            
            mdl.feed_data({'data':yuv_pad, 'img_info':image_info})
            mdl.run()

            bbox_num = mdl.get_tensor_by_name('detection_out3:1:box_out_num').to_np
            bbox = mdl.get_tensor_by_name('detection_out3:0:box_out').to_np

            bboxes, labels = post_process(bbox_num, bbox, origin_resize)

            ascend.imshow_bboxes_colors(origin_resize, bboxes, ascend.coco_classes(), colors, wait_time=10)

    del mdl

if __name__ == '__main__':
    yolov3_caffe()
