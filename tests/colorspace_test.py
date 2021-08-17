import unittest
import sys
# import HTMLTestRunner

sys.path.append("..")
import ascend
import numpy as np
import cv2
import pdb

class ColorTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("setUpClass: executed before all testcase.")

    @classmethod
    def tearDownClass(self):
        print("tearDownClass: executed after all testcase.")

    def setUp(self):
        print("execute setUp\n")

    def tearDown(self):
        print("execute tearDown")

    def assertTrue(self, expr, msg=None):
        print('[FAIL] %s' % msg if not expr else '[SUCCESS]')
        super(ColorTest, self).assertTrue(expr, msg)
    
    # all testcase write below:
    def test_imdecode_001(self):
        pdb.set_trace()
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)

        data = np.fromfile('./image/girl1.jpg', dtype=np.uint8)
        yuv = img.imdecode(data)

        # yuv_np = yuv.to_np
        # img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV21)
        # cv2.imshow('result', img_color)
        # cv2.waitKey(0)

        rgb = ascend.yuv2rgb_I420(yuv)
        img_color = rgb.to_np
        cv2.imshow('result', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.assertTrue(img_color is not None, msg="test ok")
    '''
    def test_imresize_002(self):
        pdb.set_trace()
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)

        data = np.fromfile('./image/girl1.jpg', dtype=np.uint8)
        yuv = img.imdecode(data)

        yuv_resized = img.imresize(yuv, (320, 540))
        yuv_np = yuv_resized.to_np
        img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV12)

        cv2.imshow('result', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  
        self.assertTrue(img_color is not None, msg="test ok")

    # this testcase has problem
    def test_imrescale_003(self):
        pdb.set_trace()
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)

        data = np.fromfile('./image/girl1.jpg', dtype=np.uint8)
        yuv = img.imdecode(data)

        yuv_rescale = img.imrescale(yuv, 0.3)
        yuv_np = yuv_rescale.to_np
        img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV12)
        cv2.imshow('result', img_color)
        cv2.waitKey(0)

        yuv_resized = img.imrescale(yuv, (320, 540))
        yuv_np = yuv_resized.to_np
        img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV12)
        cv2.imshow('result', img_color)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        self.assertTrue(img_color is not None, msg="test ok")

    def test_imcrop_004(self):
        pdb.set_trace()
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)

        data = np.fromfile('./image/img.jpg', dtype=np.uint8)
        yuv_img = img.imdecode(data)


        bboxes = np.array([[20, 40, 159, 259],[400, 200, 479, 419]], dtype=int)
        yuv_croped = img.imcrop(yuv_img, bboxes)
        for i, yuv in enumerate(yuv_croped):
            yuv_np = yuv.to_np
            img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV12)
            cv2.imshow('result', img_color)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        self.assertTrue(img_color is not None, msg="test ok")

    def test_imcrop_005(self):
        pdb.set_trace()
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)
        
        data = np.fromfile('./image/img.jpg', dtype=np.uint8)
        yuv_img = img.imdecode(data)

        bboxes = np.array([400, 200, 479, 419], dtype=int)
        yuv_croped = img.imcrop(yuv_img, bboxes)
        for i, yuv in enumerate(yuv_croped):
            yuv_np = yuv.to_np
            img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV12)
            cv2.imshow('result', img_color)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        self.assertTrue(img_color is not None, msg="test ok")
    

    def test_bbox_resize_006(self):
        pdb.set_trace()
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)

        data = np.fromfile('./image/img.jpg', dtype=np.uint8)
        yuv_img = img.imdecode(data)

        bboxes = np.array([[20, 40, 159, 259],[400, 200, 479, 419]], dtype=int)
        # bboxes = np.array([400, 200, 479, 419], dtype=int)
        sizes = np.array([[300, 300], [400, 400]])
        # sizes = np.array([400, 400])
        yuv_croped = img.bbox_resize(yuv_img, bboxes, sizes)
        for i, yuv in enumerate(yuv_croped):
            yuv_np = yuv.to_np
            img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV12)
            cv2.imshow('result', img_color)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        self.assertTrue(img_color is not None, msg="test ok")
    
    
    def test_impad_007(self):
        pdb.set_trace()
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)
        
        data = np.fromfile('./image/img.jpg', dtype=np.uint8)
        yuv_img = img.imdecode(data)

        yuv_resized = img.imrescale(yuv_img, (416, 416))
        img_color = cv2.cvtColor(yuv_resized.to_np, cv2.COLOR_YUV2RGB_NV21)
        cv2.imshow('result', img_color)
        cv2.waitKey()

        yuv_pad = img.impad(yuv_resized, shape=(416, 416))
        yuv_pad_np = yuv_pad.to_np
        img_color = cv2.cvtColor(yuv_pad_np, cv2.COLOR_YUV2RGB_NV21)
        cv2.imshow('result', img_color)
        cv2.waitKey()

        cv2.destroyAllWindows()
        self.assertTrue(img_color is not None, msg="test ok")
    
    def test_impad_008(self):
        context = ascend.Context({0})
        ctx = context.context_dict[0]
        img = ascend.Image(ctx)
        
        data = np.fromfile('./image/img.jpg', dtype=np.uint8)
        yuv_img = img.imdecode(data)

        pdb.set_trace()
        yuv_resized = img.imrescale(yuv_img, (416, 416))
        img_color = cv2.cvtColor(yuv_resized.to_np, cv2.COLOR_YUV2RGB_NV21)
        cv2.imshow('result1', img_color)
        cv2.waitKey()

        yuv_pad = img.impad(yuv_resized, padding=(20, 50, 100, 200), pad_val=128)
        yuv_pad_np = yuv_pad.to_np
        img_color2 = cv2.cvtColor(yuv_pad_np, cv2.COLOR_YUV2RGB_NV21)
        cv2.imshow('result2', img_color2)
        cv2.waitKey()

        cv2.destroyAllWindows()
        self.assertTrue(img_color is not None, msg="test ok")
    
    # def test_imcrop_paste_009(self):
    #     import pdb
    #     context = ascend.Context({0})
    #     ctx = context.context_dict[0]
    #     img = ascend.Image(ctx)
        
    #     src_img = np.fromfile('./image/xiaoxin.jpg', dtype=np.uint8)
    #     yuv_src = img.imdecode(src_img)

    #     dst_img = np.fromfile('./image/img.jpg', dtype=np.uint8)
    #     yuv_dst = img.imdecode(dst_img)

    #     pdb.set_trace()
    #     crop_bbox = np.array([40, 30, 140, 230], dtype='int32')
    #     paste_bbox = np.array([70, 80, 170, 280], dtype='int32')
    #     img.imcrop_paste(yuv_src, yuv_dst, crop_bbox, paste_bbox)
    #     img_color = cv2.cvtColor(yuv_dst.to_np, cv2.COLOR_YUV2RGB_NV21)
    #     cv2.imshow('result1', img_color)
    #     cv2.waitKey()

    #     crop_bbox = np.array([[32, 64, 200, 180], [400, 320, 520, 487]], dtype='int32')
    #     paste_bbox = np.array([[80, 73, 193, 240], [46, 320, 238, 640]], dtype='int32')
    #     img.imcrop_paste(yuv_src, yuv_dst, crop_bbox, paste_bbox)
    #     img_color = cv2.cvtColor(yuv_dst.to_np, cv2.COLOR_YUV2RGB_NV21)
    #     cv2.imshow('result2', img_color)
    #     cv2.waitKey()

    #     cv2.destroyAllWindows()
    #     self.assertTrue(img_color is not None, msg="test ok")
    '''

if __name__ == '__main__':
        #####################################
        # 1.test single case
        # ImageTest is the object name, test_TS_001 is the case name
        # suite = unittest.TestSuite()
        # suite.addTest(ImageTest("test_TS_001"))
        # suite.addTest(ImageTest("test_TS_002"))
        # runner = unittest.TextTestRunner().run(suite)

        # save the test result to html
        # filename = './apptestresult.html'
        # fb = open(filename, 'wb')
        # runner = HTMLTestRunner.HTMLTestRunner(stream=fb, title="测试HTMLTestRunner", description="测试HTMLTestRunner")
        # runner.run(suite)
        # fb.close()

        ######################################
        # 2. test all case
        unittest.main(testRunner=unittest.TextTestRunner(stream=None, verbosity=2))

