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


class YOLOV5():
    def __init__(self, context, model_path, confThresh=0.5, nmsThresh=0.5):
        # initial model
        self.model = ascend.AscendModel(context, model_path)
        self.classes = ascend.coco_classes()

        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.layer_num = 3
        self.top_k = 300  # maximum number of detections per image
        self.no = 85
        self.stride = np.array([8, 16, 32])
        self.anchor_grid = np.array(anchors).reshape(3, 1, 3, 1, 1, 2)

        self.conf_th = confThresh
        self.nms_th = nmsThresh

    def _make_grid(self, nx=20, ny=20):
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((x, y), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def _sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

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
        self.model.feed_data({'images':src_img})
        t = time.time()
        self.model.run()
        print(time.time() - t)

        # get feature map of three output layer
        tensor1 = self.model.get_tensor_by_name('Transpose_271:0').to_np # 80 * 80
        tensor2 = self.model.get_tensor_by_name('Transpose_287:0').to_np
        tensor3 = self.model.get_tensor_by_name('Transpose_303:0').to_np

        z, grid = [], []
        for i, tensor in enumerate((tensor1, tensor2, tensor3)):
            # make grid according feature map's shape
            _, _, ny, nx, _ = tensor.shape
            grid.append(self._make_grid(nx, ny))

            # do sigmoid activation
            y = self._sigmoid(tensor)

            # caculate location x,y and shift w,h
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(1, -1, self.no))
        return np.concatenate(z, 1)

    def xywh2xyxy(self, pred):
        """ Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] 
            where: top left point (x1, y1), bottom right point (x2, y2)
                x1 = center_x - w/2, 
                y1 = center_y - h/2.
                x2 = center_x + w/2, 
                y2 = center_y + w/2.
        Args:
            pred (ndarray): input predition location.

        Returns:
            transformed result (ndarray).
        """
        y = np.zeros_like(pred)
        y[:, 0] = pred[:, 0] - pred[:, 2] / 2  # top left x
        y[:, 1] = pred[:, 1] - pred[:, 3] / 2  # top left y
        y[:, 2] = pred[:, 0] + pred[:, 2] / 2  # bottom right x
        y[:, 3] = pred[:, 1] + pred[:, 3] / 2  # bottom right y
        return y

    def scale_coords(self, mdl_shape, coords, img_shape):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        scale = max(img_shape[0] / mdl_shape[0], img_shape[1] / mdl_shape[1]) 
        coords[:, :4] *= scale

        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, 0] = coords[:, 0].clip(0, img_shape[1])  # x1
        coords[:, 1] = coords[:, 1].clip(0, img_shape[0])  # y1
        coords[:, 2] = coords[:, 2].clip(0, img_shape[1])  # x2
        coords[:, 3] = coords[:, 3].clip(0, img_shape[0])  # y2
        return coords


    def post_process(self, pred, mdl_shape, image_shape):
        class_num = pred[0].shape[1] - 5  # number of classes
        conf_mask = pred[..., 4] > self.conf_th  # candidates

        # Process detections
        res = []
        for batch_i, det in enumerate(pred):  # detections per image
            if det is None:
                continue

            x = det[conf_mask[batch_i]]  # confidence

            # compute conf with method: conf = obj_conf * cls_conf
            x[:, 5:] *= x[:, 4:5]  

            # transform bbox(center_x, center_y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            i, j = np.nonzero(x[:, 5:] > self.conf_th)
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype('float32')), 1)

            # Batched NMS
            boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
            idx = ascend.nms(boxes, scores, self.nms_th)[:self.top_k]

            det = x[idx]
            # Rescale boxes from img_size to im0 size
            det[:, :4] = self.scale_coords(mdl_shape, det[:, :4], image_shape).round()
            res.append(det)
        return res

    def __del__(self):
        del self.model


def yolov5_detection():
    device_id = 0
    model_path = "../tools/model/yolov5s_sim_t_fp32_aipp.om"
    video_stream_path = '/home/zhengguojian/ascendfly/tests/video/test-1k.264'
    # video_stream_path = '/home/zhengguojian/demo/tests/video/changsha_wuyi_road.264'
    ctx = ascend.Context({device_id}).context_dict[device_id]
    Img = ascend.Image(ctx)
    cap = ascend.VideoCapture(ctx, video_stream_path)
    yolo = YOLOV5(ctx, model_path)

    colors = ascend.color_gen(len(ascend.coco_classes()))
    while cap.is_open():
        yuv_img, frame_id = cap.read()
        if yuv_img:
            yuv_resized = Img.imrescale(yuv_img, (640, 640))
            yuv_pad = Img.impad(yuv_resized, shape=(640, 640))
            origin_img = cv2.cvtColor(yuv_resized.to_np, cv2.COLOR_YUV2RGB_NV21)

            t1 = time.time()
            pred = yolo.detect(yuv_pad)
            bboxes = yolo.post_process(pred, (640, 640), origin_img.shape)
            print("elapse: ", time.time() - t1)

            ascend.imshow_bboxes_colors(origin_img, bboxes[0], ascend.coco_classes(), colors, wait_time=25)

if __name__ == '__main__':
    yolov5_detection()
    