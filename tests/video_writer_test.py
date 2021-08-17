import unittest
import sys

sys.path.append("..")
import ascend
import numpy as np
import cv2
import pdb

class VideWriterTest(unittest.TestCase):
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
        super(VideWriterTest, self).assertTrue(expr, msg)


    def test_VideoWriter_001(self):
        stream_dir = './test_venc_h264.264'
        context = ascend.Context({1}).context_dict[1]
        Img = ascend.Image(context)

        video_stream_path = './video/cars_around_mountain_640_360.264'
        cap = ascend.VideoCapture(context, video_stream_path)

        pdb.set_trace()
        encode = ascend.VideoWriter(context, stream_dir, 25, (640, 480))
        ascend.show_growth()

        while cap.is_open():
            image, frame_id = cap.read()
            if image:
                print(f"process frame:{frame_id}")
                yuv_resize = Img.imresize(image, (640, 480))
            
                encode.write(yuv_resize)

        encode.release()
        cap.release()
        
        del cap
        del Img

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

