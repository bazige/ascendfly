import unittest
import sys
# import HTMLTestRunner

sys.path.append("..")
import ascend
import numpy as np
import cv2
import pdb

class OpTest(unittest.TestCase):
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
        super(OpTest, self).assertTrue(expr, msg)

    '''
    def test_cast_fp16_fp32_001(self):
        resource = ascend.Context({0})
        data = np.arange(3*32*32, dtype='float32')
        data = data.reshape((1, 3, 32, 32))
        ascend_data = ascend.AscendArray.clone(data)
        
        out = ascend.Cast(ascend_data, dtype=np.dtype('float16'))
        out = ascend.Cast(out.data, dtype=np.dtype('float32'))
        
        ret = np.allclose(out.data.to_np, data, rtol=1e-03, atol=1e-03, equal_nan=False)
        self.assertTrue(ret, msg="test_cast_fp16_fp32_001 ok")
        
    
    def test_cast_uint8_fp16_002(self):
        resource = ascend.Context({0})
        data = np.arange(25*25, dtype='uint8')
        data = data.reshape((1, 1, 25, 25))
        ascend_data = ascend.AscendArray.clone(data)

        out = ascend.Cast(ascend_data, dtype=np.dtype('float16'))
        out = ascend.Cast(out.data, dtype=np.dtype('uint8'))

        ori = data.astype(np.float16).astype(np.uint8)
        res = np.allclose(out.data.to_np, ori, rtol=1e-03, atol=1e-03, equal_nan=False)
        self.assertTrue(res, msg="test_cast_uint8_fp16_002 ok")

    
    def test_cast_int32_fp16_003(self):
        resource = ascend.Context({0})
        shape = (1, 16, 16, 2)
        data = np.arange(int(np.prod(shape)), dtype='int32')
        data = data.reshape(shape)
        ascend_data = ascend.AscendArray.clone(data)
        pdb.set_trace()

        out = ascend.Cast(ascend_data, dtype=np.dtype('float16'))
        out = ascend.Cast(out.data, dtype=np.dtype('int32'))
        ori = data.astype(np.float16).astype(np.int32)
        res = np.allclose(out.data.to_np, ori, rtol=1e-03, atol=1e-03, equal_nan=False)
        self.assertTrue(res, msg="test_cast_int32_fp16_003 ok")
    
 
    def test_transpose_004(self):
        resource = ascend.Context({0})
        shape = (1, 64, 64)
        data = np.arange(int(np.prod(shape)), dtype='float16')
        data = data.reshape(shape)
        ascend_data = ascend.AscendArray.clone(data)

        pdb.set_trace()
        out = ascend.Transpose(ascend_data, perm=[1, 2, 0])
        ori = np.transpose(data, axes=[1, 2, 0])
        res = np.allclose(out.data.to_np, ori, rtol=1e-03, atol=1e-03, equal_nan=False)
        self.assertTrue(res, msg="test_transpose_004 ok")
    
    def test_cast_005():
        resource = ascend.Context({12})
        data = np.arange(25, dtype='uint8')
        data = data.reshape((5, 5))
        ascend_data = ascend.AscendArray.clone(data)
        print(ascend_data[:])
        pdb.set_trace()
        out = ascend.Cast(ascend_data, dtype=np.dtype('float16'))
        print(out[:])
        del ascend_data
    
    def test_argmaxd_006(self):
        resource = ascend.Context({0})
  
        pdb.set_trace()
        data = np.random.rand(100).astype(np.float16)
        ascend_data = ascend.AscendArray.clone(data)
        out = ascend.ArgMax(ascend_data)
        res = np.allclose(out.data.to_np, np.argmax(data), rtol=1e-03, atol=1e-03, equal_nan=False)
        # self.assertTrue(res, msg="test_transpose_004 ok")
        pdb.set_trace()
        data = np.random.rand(128, 243).astype(np.float16)
        ascend_data = ascend.AscendArray.clone(data)
        out = ascend.ArgMax(ascend_data)
        res = np.allclose(out.data.to_np, np.argmax(data), rtol=1e-03, atol=1e-03, equal_nan=False)
        # self.assertTrue(res, msg="test_transpose_004 ok")

        data = np.random.rand(8, 3, 324, 324).astype(np.float16)
        ascend_data = ascend.AscendArray.clone(data)
        out = ascend.ArgMax(ascend_data)
        res = np.allclose(out.data.to_np, np.argmax(data), rtol=1e-03, atol=1e-03, equal_nan=False)
        self.assertTrue(res, msg="test_argmaxd_005 ok")

    '''
    def test_permute_007(self):
        resource = ascend.Context({0})
  
        pdb.set_trace()
        data = np.random.rand(1, 3, 324, 324).astype(np.float16)
        ascend_data = ascend.AscendArray.clone(data)
        out = ascend.Permute(ascend_data)
        res = np.allclose(out.data.to_np, np.transpose(data, (0, 2, 3, 1)), rtol=1e-03, atol=1e-03, equal_nan=False)
        self.assertTrue(res, msg="test_permute_007 ok")


        data = np.random.rand(1, 324, 324, 3).astype(np.float16)
        ascend_data = ascend.AscendArray.clone(data)
        out = ascend.Permute(ascend_data, axis=(0, 3, 1, 2))
        res = np.allclose(out.data.to_np, np.transpose(data, (0, 3, 1, 2)), rtol=1e-03, atol=1e-03, equal_nan=False)
        self.assertTrue(res, msg="test_permute_007 ok")


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

