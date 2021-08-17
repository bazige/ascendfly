import unittest
# import HTMLTestRunner
import sys

sys.path.append("..")
import ascend
import numpy as np
import cv2
import pdb

class VdecTest(unittest.TestCase):
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
        super(VdecTest, self).assertTrue(expr, msg)

    # # all testcase write below:
    # def test_VideoCapture_001(self):
    #     import os
    #     resource = ascend.Context({0})
    #     video_stream_path = os.getcwd() + '/data/cars_around_mountain_640_360.264'

    #     context = resource.context_dict[0]
    #     cap = ascend.VideoCapture(context, video_stream_path)

    #     while cap.is_open():
    #         image, frame_id = cap.read()
    #         if image:
    #             yuv_np = image.to_np
    #             img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV21)
    #             cv2.imshow('result', img_color)
    #             cv2.waitKey(10)
    #     cv2.destroyAllWindows()

    #     self.assertTrue(True, msg="test ok")
    #     del cap
    #     del resource
    
    # def test_VideoCapture_002(self):
    #     import os
    #     resource = ascend.Context({0})
    #     video_stream_path = os.getcwd() + '/data/cars_around_mountain_640_360.264'

    #     ctx = resource.context_dict[0]
    #     cap = ascend.VideoCapture(ctx, video_stream_path)

    #     # do model inference
    #     pdb.set_trace()
    #     model_path = "modelzoo/centerface_static_aipp_nv12.om"
    #     model = ascend.AscendModel(ctx, model_path)
    #     print(model.tensor)
    #     img = ascend.Image(ctx)
    #     while cap.is_open():
    #         image, frame_id = cap.read()
    #         if image:
    #             # pdb.set_trace()
    #             yuv_resized = img.imresize(image, (384, 384))
    #             model.feed_data({'data':yuv_resized})

    #             model.run()

    #             ascend_out1 = model.get_tensor_by_name('537:0:537')
    #             out1 = ascend_out1.to_np
    #             print(out1)

    #             ascend_out2 = model.get_tensor_by_name('538:0:538')
    #             out2 = ascend_out2.to_np
    #             print(out2)

    #             yuv_np = yuv_resized.to_np
    #             img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV21)
    #             cv2.imshow('result', img_color)
    #             cv2.waitKey(5)

    #     cv2.destroyAllWindows()

    #     self.assertTrue(True, msg="test ok")

    def test_VideoCapture_001(self):
        import os
        resource = ascend.Context({0})
        video_stream_path = './test-1k.264'

  
        context = resource.context_dict[0]
        cap = ascend.VideoCapture(context, video_stream_path)

        model_path = "../tools/model/centerface_static_aipp_nv12.om"
        model = ascend.AscendModel(context, model_path)
        img = ascend.Image(context)

        while cap.is_open():
            image, frame_id = cap.read()
            if image:
                yuv_np = image.to_np
                img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV21)
                # cv2.imshow('result', img_color)
                # cv2.waitKey(10)
                yuv_resized = img.imresize(image, (384, 384))
                model.feed_data({'data':yuv_resized})
                model.run()
                ascend_out1 = model.get_tensor_by_name('537:0:537')
                ascend_out2 = model.get_tensor_by_name('538:0:538')
                out1 = ascend_out1.to_np
                out2 = ascend_out2.to_np

        # cv2.destroyAllWindows()

        self.assertTrue(True, msg="test ok")
        del cap
        del resource

if __name__ == '__main__':
        #####################################
        # 1.test single case
        # VdecTest is the object name, test_TS_001 is the case name
        # suite = unittest.TestSuite()
        # suite.addTest(VdecTest("test_TS_001"))
        # suite.addTest(VdecTest("test_TS_002"))
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

