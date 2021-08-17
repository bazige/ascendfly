import unittest
import sys

sys.path.append("..")
import ascend
import numpy as np
import cv2

class AscendArrayTest(unittest.TestCase):
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
        super(AscendArrayTest, self).assertTrue(expr, msg)

    def test_ascendarray_script_000(self):
        resource = ascend.Context({12})
        arr = np.arange(24, dtype='float32')
        arr = arr.reshape((4, 6))
        ascend_arr = ascend.AscendArray.clone(arr)

        print(np.isclose(arr, ascend_arr[:], rtol=1e-5, atol=1e-5).all())
        print(np.isclose(arr[:3], ascend_arr[:3], rtol=1e-5, atol=1e-5).all())
        print(np.isclose(arr[2][3], ascend_arr[2][3], rtol=1e-5, atol=1e-5))
        print(np.isclose(arr[2:4], ascend_arr[2:4], rtol=1e-5, atol=1e-5).all())

        array = np.ones(shape=(384, 384), dtype='float32')
        ascend_array = ascend.AscendArray.clone(array)
        print(ascend_array[2])

        print(np.isclose(array[:32], ascend_array[:32], rtol=1e-5, atol=1e-5).all())
        del resource

    def test_ascend_slice_001(self):
        import pdb
        pdb.set_trace()
        resource = ascend.Context({12})
        data = np.arange(25, dtype='uint8')
        data = data.reshape((5, 5))
        ascend_data = ascend.AscendArray.clone(data)
        print(ascend_data[0][3])
        ascend_data[0][3] = 10
        data2 = ascend_data.to_np
        data2 = data2.reshape((5, 5))
        print(data2[0][3])
        # assert data2[0][3] == 10

        ascend_data[1] = [3, 3, 3, 3, 3]
        data2 = ascend_data.to_np
        data2 = data2.reshape((5, 5))
        print(np.isclose(data2[1], np.array([3, 3, 3, 3, 3], dtype='uint8'), rtol=1e-5, atol=1e-5).all())

        del resource

    def test_ascend_astype_002(self):
        resource = ascend.Context({12})
        data = np.arange(32*32*3, dtype='uint8')
        data = data.reshape((1, 32, 32, 3))

        ascend_data = ascend.AscendArray.clone(data)
        print(ascend_data[:])

        out1 = ascend_data.astype(np.float16)
        out2 = out1.astype(np.float32)
        print(ascend_data[:])
     
   
    def test_ascend_to_np_003(self):
        import pdb
        pdb.set_trace()
        resource = ascend.Context({0})
        data = np.arange(256*3, dtype='float16')
        data = data.reshape((3, 16, 16))
        ascend_data = ascend.AscendArray.clone(data)
        print(np.isclose(data, ascend_data.to_np, rtol=1e-3, atol=1e-3).all())

        data = np.random.random(8*23*256).reshape(8, 23, 256).astype('float32')
        ascend_data = ascend.AscendArray.clone(data)
        print(np.isclose(data, ascend_data.to_np, rtol=1e-5, atol=1e-5).all())
       
        data = np.random.random(8*23*256).reshape(8, 23, 256)*256
        data = data.astype('int32')
        ascend_data = ascend.AscendArray.clone(data)
        print(np.isclose(data, ascend_data.to_np, rtol=1e-3, atol=1e-3).all())


        data = np.random.randint(-32768, 32768, size=5*43*82).reshape(5, 43, 82)
        data = data.astype('int32')
        ascend_data = ascend.AscendArray.clone(data)
        print(np.isclose(data, ascend_data.to_np, rtol=1e-3, atol=1e-3).all())

        del ascend_data

if __name__ == '__main__':
        #####################################
        # 1.test single case
        # ImageTest is the object name, test_TS_001 is the case name
        suite = unittest.TestSuite()
        # suite.addTest(AscendArrayTest("test_ascendarray_script_000"))
        # suite.addTest(AscendArrayTest("test_ascend_slice_001"))
        suite.addTest(AscendArrayTest("test_ascend_astype_002"))
        # suite.addTest(AscendArrayTest("test_ascend_to_np_003"))
        runner = unittest.TextTestRunner().run(suite)

        # save the test result to html
        # filename = './apptestresult.html'
        # fb = open(filename, 'wb')
        # runner = HTMLTestRunner.HTMLTestRunner(stream=fb, title="测试HTMLTestRunner", description="测试HTMLTestRunner")
        # runner.run(suite)
        # fb.close()

        ######################################
        # 2. test all case
        # unittest.main(testRunner=unittest.TextTestRunner(stream=None, verbosity=2))