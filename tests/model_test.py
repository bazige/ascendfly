import unittest
# import HTMLTestRunner
import sys

sys.path.append("..")
import ascend
import numpy as np
import cv2
import os


def decode(heatmap, scale, offset, landmark, size, thresh=0.1):
    heatmap = np.squeeze(heatmap[0])
    scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
    offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
    c0, c1 = np.where(heatmap > thresh)
    
    boxes, lms = [], []
    if len(c0) > 0:
        for i in range(len(c0)):
            s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
            o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
            s = heatmap[c0[i], c1[i]]
            x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
            x1, y1 = min(x1, size[1]), min(y1, size[0])
            boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
            
            lm = []
            for j in range(5):
                lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
            lms.append(lm)
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = ascend.nms(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :]
   
        lms = np.asarray(lms, dtype=np.float32)
        lms = lms[keep, :]
    return boxes, lms


def post_process(heatmap, scale, offset, lms, thresh=0.3, scale_shape=None, with_landmark=False):
    w, h, scale_w, scale_h = scale_shape
    dets, lms = decode(heatmap, scale, offset, lms, (h, w), thresh=thresh)
   
    if len(dets) > 0:
        dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / scale_w, dets[:, 1:4:2] / scale_h
        lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / scale_w, lms[:, 1:10:2] / scale_h
    else:
        dets = np.empty(shape=[0, 5], dtype=np.float32)
        lms = np.empty(shape=[0, 10], dtype=np.float32)
        
    if with_landmark:
        return dets, lms
    else:
        return dets

class InferTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("setUpClass: executed before all testcase.")

    @classmethod
    def tearDownClass(self):
        print("tearDownClass: executed after all testcase.")

    def setUp(self):
        print("execute setUp")

    def tearDown(self):
        print("execute tearDown")

    def assertTrue(self, expr, msg=None):
        print('[FAIL] %s' % msg if not expr else '[SUCCESS]')
        super(InferTest, self).assertTrue(expr, msg)
    
    # all testcase write below:
    def test_rawdata_infer_001(self):
        resource = ascend.Context({0})
        model_path = "../tools/model/centerface_noaipp.om"
        array = np.ones(shape=(1, 3, 384, 384), dtype='float32')
        dev_data = ascend.AscendArray(shape=(1, 3, 384, 384), dtype=np.dtype('float32'))
        dev_data.to_ascend(array)
        print(dev_data.to_np)
        dev_data2 = ascend.AscendArray.clone(array)

        for _, ctx in resource.context_dict.items():
            model = ascend.AscendModel(ctx, model_path)
            model.feed_data({'data':dev_data2})
            model.run()
            ascend_out = model.get_tensor_by_name('537:0:537')
            out = ascend_out.to_np
            print(ascend_out.to_np)
            print(out.data)
            print(out.dtype)
            print(out.shape)
            print(out.nbytes)
            print(out.itemsize)

            ascend_out = model.get_tensor_by_name('538:0:538')
            out = ascend_out.to_np
            print(ascend_out.to_np)
            print(out.data)
            print(out.dtype)
            print(out.shape)
            print(out.nbytes)
            print(out.itemsize)

        self.assertTrue(out is not None, msg="test ok")
        del model
        del resource
    
    def test_rawdata_infer_002(self):
        resource = ascend.Context({0})
        model_path = "../tools/model/centerface_noaipp.om"
        array = np.ones(shape=(3, 384, 384), dtype='float32')
        dev_data2 = ascend.AscendArray.clone(array)

        for _, ctx in resource.context_dict.items():
            model = ascend.AscendModel(ctx, model_path)
            model.feed_data({'data':dev_data2})
            model.run()
            ascend_out = model.get_tensor_by_name('537:0:537')
            print(ascend_out.to_np)

            ascend_out = model.get_tensor_by_name('538:0:538')
            print(ascend_out.to_np)

        self.assertTrue(True, msg="test ok")
        del model
        del resource
    
    def test_rawdata_infer_003(self):
        resource = ascend.Context({0})
        model_path = "../tools/model/centerface_static_aipp_nv12.om"
        array = np.ones(shape=(int(384*1.5), 384), dtype='float32')
        dev_data = ascend.AscendArray(shape=(int(384*1.5), 384), dtype=np.dtype('float32'))
        dev_data.to_ascend(array)
        print(dev_data.to_np)

        for _, ctx in resource.context_dict.items():
            model = ascend.AscendModel(ctx, model_path)
            
            model.feed_data({'data':dev_data})
            model.run()
            ascend_out = model.get_tensor_by_name('537:0:537')
            print(ascend_out.to_np)

            ascend_out = model.get_tensor_by_name('538:0:538')
            print(ascend_out.to_np)
    

        self.assertTrue(True, msg="test ok")
        del model
        del resource
    
    def test_video_infer_004(self):
        resource = ascend.Context({0})
        model_path = "../tools/model/centerface_static_aipp_nv12.om"
        video_stream_path = os.getcwd() + '/video/test-1k.264'
        landmarks = False
        ctx = resource.context_dict[0]
        Img = ascend.Image(ctx)

        cap = ascend.VideoCapture(ctx, video_stream_path)
        mdl = ascend.AscendModel(ctx, model_path)
        while cap.is_open():
            yuv_img, frame_id = cap.read()
            if yuv_img:

                yuv_resized = Img.imrescale(yuv_img, (384, 384))
                yuv_pad = Img.impad(yuv_resized, shape=(384, 384))
                img_color = cv2.cvtColor(yuv_resized.to_np, cv2.COLOR_YUV2RGB_NV21)
                
                mdl.feed_data({'data':yuv_pad})
                mdl.run()

                heatmap = mdl.get_tensor_by_name('537:0:537').to_np
                scale = mdl.get_tensor_by_name('538:0:538').to_np
                offset = mdl.get_tensor_by_name('539:0:539').to_np
                lms = mdl.get_tensor_by_name('540:0:540').to_np
   
                dets = post_process(heatmap, scale, offset, lms, thresh=0.3, scale_shape=(384, 384, 1.0, 1.0))
                
                for det in dets:   
                    boxes, score = det[:4], det[4]
                    cv2.rectangle(img_color, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
                if landmarks:
                    for lm in lms:
                        for i in range(0, 5):
                            cv2.circle(img_color, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
                cv2.imshow('out', img_color)
                cv2.waitKey(25)

        self.assertTrue(True, msg="test ok")
        del mdl
        del resource
    
    def test_batch_image_infer_005(self):
        resource = ascend.Context({0})
        model_path = "../tools/model/centerface_8batch_static_aipp_nv12.om"
        video_stream_path = os.getcwd() + '/video/test-1k.264'

        landmarks = False
        ctx = resource.context_dict[0]
        Img = ascend.Image(ctx)
        cap = ascend.VideoCapture(ctx, video_stream_path)
        mdl = ascend.AscendModel(ctx, model_path)

        count = 0
        imgs = []
        while cap.is_open():
            yuv_img, frame_id = cap.read()
            if yuv_img:
                yuv_show = Img.imrescale(yuv_img, (1280, 720))
                yuv_resized = Img.imrescale(yuv_img, (384, 384))
                yuv_pad = Img.impad(yuv_resized, shape=(384, 384))
                if count != mdl.tensor['data'].shape[0]:
                    imgs.append(yuv_pad)
                    count = count + 1
                    continue
                
                in_data = ascend.imgs2tensor(imgs, tensor_fmt='NHWC')
                mdl.feed_data({'data':yuv_pad})
                mdl.run()
                count = 0
                imgs = []

                heatmap = mdl.get_tensor_by_name('537:0:537').to_np
                scale = mdl.get_tensor_by_name('538:0:538').to_np
                offset = mdl.get_tensor_by_name('539:0:539').to_np
                lms = mdl.get_tensor_by_name('540:0:540').to_np
   
                scale_shape = yuv_show.shape + (384/max(yuv_show.shape), 384/max(yuv_show.shape))
                dets = post_process(heatmap, scale, offset, lms, thresh=0.3, scale_shape=scale_shape)
                
                ascend.show_bbox(yuv_show, dets, color=(0,255,0) , thickness=1, wait_time_ms=5)

        self.assertTrue(True, msg="test ok")
        del mdl
        del resource
    
    def test_image_infer_006(self):
        resource = ascend.Context({0})
        model_path = "../tools/model/centerface_static_aipp_nv12.om"

        for _, ctx in resource.context_dict.items():
            # image decode and resize
            img = ascend.Image(ctx)
            data = np.fromfile('./image/girl1.jpg', dtype=np.uint8)
            yuv = img.imdecode(data)
            yuv_resized = img.imresize(yuv, (384, 384))

            # do model inference
            model = ascend.AscendModel(ctx, model_path)
            model.feed_data({'data':yuv_resized})
            model.run()
            heatmap = model.get_tensor_by_name('537:0:537').to_np
            scale = model.get_tensor_by_name('538:0:538').to_np
            offset = model.get_tensor_by_name('539:0:539').to_np
            lms = model.get_tensor_by_name('540:0:540').to_np

            scale_shape = yuv.shape + (384/max(yuv.shape), 384/max(yuv.shape))
            dets = post_process(heatmap, scale, offset, lms, thresh=0.3, scale_shape=scale_shape)

            ascend.show_bbox(yuv, dets, color=(0,255,0) , thickness=2, wait_time_ms=0)
        self.assertTrue(dets is not None, msg="test ok")
        del model
        del resource
    
    
    def test_model_profiling_007(self):
        context = ascend.Context({1})
        model_path = "../tools/model/yolov5s_bs1_fp16.om"
        array = np.ones(shape=(1, 3, 640, 640), dtype='float16')
        data = ascend.AscendArray.clone(array)

        for ctx in context:
            model = ascend.AscendModel(ctx, model_path)
            model.feed_data({'images':data})
            print(model.tensor_names)
   
            prof = ascend.Profiling(ctx, model.model_id)

            @prof.elapse_time
            @prof.profiling
            def run():
                model.run()

            run()
            prof.info_print(sort=True)

            ascend_out = model.get_tensor_by_name('Transpose_271:0')
            out = ascend_out.to_np
            # print(ascend_out.to_np)
            # print(out.data)

        self.assertTrue(out is not None, msg="test ok")
        del model
        del context
    
    def test_mul_device_008(self):
        context = ascend.Context({0, 1})
        model_path1 = "../tools/model/BERT_text.om"
        model_path2 = "../tools/model/centerface_static_aipp_nv12.om"
        
        model1 = ascend.AscendModel(context.context_dict[0], model_path1)
        array = np.ones(shape=(1, 512), dtype='int32')
        data = ascend.AscendArray.clone(array)
        model1.feed_data({'input_ids':data, 'input_mask':data})
        ascend_out = model1.get_tensor_by_name('item_embedding:0')
        print(ascend_out.to_np)


        model2 = ascend.AscendModel(context.context_dict[1], model_path2)
        img = ascend.Image(context.context_dict[1])
        data = np.fromfile('./image/girl1.jpg', dtype=np.uint8)
        yuv = img.imdecode(data)
        yuv_resized = img.imresize(yuv, (384, 384))
        model.feed_data({'data':yuv_resized})
        model.run()
        heatmap = model.get_tensor_by_name('537:0:537').to_np
        scale = model.get_tensor_by_name('538:0:538').to_np
        offset = model.get_tensor_by_name('539:0:539').to_np
        lms = model.get_tensor_by_name('540:0:540').to_np

        scale_shape = yuv.shape + (384/max(yuv.shape), 384/max(yuv.shape))
        dets = post_process(heatmap, scale, offset, lms, thresh=0.3, scale_shape=scale_shape)

        ascend.show_bbox(yuv, dets, color=(0,255,0) , thickness=2, wait_time_ms=0)



if __name__ == '__main__':
        #####################################
        # 1.test single case
        # InferTest is the object name, test_TS_001 is the case name
        suite = unittest.TestSuite()
        # suite.addTest(InferTest("test_rawdata_infer_001"))
        # suite.addTest(InferTest("test_rawdata_infer_002"))
        # suite.addTest(InferTest("test_rawdata_infer_003"))
        # suite.addTest(InferTest("test_video_infer_004"))
        # suite.addTest(InferTest("test_batch_image_infer_005"))
        # suite.addTest(InferTest("test_image_infer_006"))
        suite.addTest(InferTest("test_model_profiling_007"))

        # suite.addTest(InferTest("test_model_profiling_002"))
        runner = unittest.TextTestRunner().run(suite)

        ######################################
        # 2. test all case
        # unittest.main(testRunner=unittest.TextTestRunner(stream=None, verbosity=2))

