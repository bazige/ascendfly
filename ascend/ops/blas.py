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
import pdb
import acl
import numpy as np

from ..common.const import *
from ..data.ascendarray import AscendArray
from ..resource.context import create_stream

ACL_TRANS_N = 0 

def type_map(data_type):
    type_dict = {
        np.dtype('float32'):ACL_FLOAT,
        np.dtype('float16'):ACL_FLOAT16,
        np.dtype('int8')   :ACL_INT8
    }

    try:
        return type_dict[data_type]
    except KeyError:
        raise TypeError(f"Input data_type expects a fp16 or int8, but got {type(data_type)}.")


class Matmul():
    """ define a Matmul object, release the function .

    Attributes::
        context: the output image bind with an AscendArray object, image.shape(tupe(h, w, c))
        mat_a  : 
        mat_b  : 
        mat_c  :

    Methods:
        __pre_compute : prepare input data
        run           : do compute matmul
        out           : return output result
    """
    def __init__(self, mat_a, mat_b, mat_c, alpha=1.0, beta=0.0, highprec=True, context=None):
        if context and not isinstance(context, int):
            raise TypeError(f"Input context expects an int, but got {type(context)}.")

        if not isinstance(mat_a, AscendArray):
            raise TypeError(f"Input mat_a expects an AscendArray, but got {type(mat_a)}.")

        if not isinstance(mat_b, AscendArray):
            raise TypeError(f"Input mat_a expects an AscendArray, but got {type(mat_b)}.")

        if not isinstance(mat_c, AscendArray):
            raise TypeError(f"Input mat_a expects an AscendArray, but got {type(mat_c)}.")

        assert mat_a.format == mat_b.format, f"Input mat_a and mat_b expects same format."

        # assign self value
        self.context = context
        self.stream = create_stream(context)
        self.highprec = 1 if highprec else 0

        pdb.set_trace()
        # set op model dir
        ret = acl.op.set_model_dir("./om")
        if ret != ACL_SUCCESS:
            raise ValueError(f"op set model dir failed, return {ret}.")
        
        # calculate m, n, k and trans alpha and beta to np.ndarray
        self.__pre_compute(mat_a, mat_b, mat_c, alpha, beta)

        # do blas gemm_ex and synchronize stream
        self.run()

        # free input data memory
        self.free_ab()


    def __pre_compute(self, mat_a, mat_b, mat_c, alpha, beta):
        """ calculate m, n, k and copy alpha/beta to device.
        Args:
            mat_a : (AscendArray) matrix A
            mat_b : (AscendArray) matrix B
            mat_c : (AscendArray) matrix C
            alpha : (float value)
            beta  : (float value)

        Returns:
            None
        """
        if mat_a.format in [
            PIXEL_FORMAT_YUV_SEMIPLANAR_420,
            PIXEL_FORMAT_YVU_SEMIPLANAR_420
            ]:
            self.m = mat_a.shape[0] * 2 // 3
            self.n = mat_b.shape[-1]
            self.k = mat_a.shape[1]
        elif mat_a.format in [
            PIXEL_FORMAT_RGB_888,
            PIXEL_FORMAT_BGR_888
            ]:
            self.m = mat_a.shape[0]
            self.n = mat_b.shape[-1]
            self.k = mat_a.shape[1]
        elif mat_a.format == 'NCHW':
            self.m = mat_a.shape[2]
            self.n = mat_b.shape[-1]
            self.k = mat_a.shape[-1]
        elif mat_a.format == 'NHWC':
            self.m = mat_a.shape[1]
            self.n = mat_b.shape[-1]
            self.k = mat_a.shape[2]
        else:
            raise ValueError(f"Input data format not support.")

        alpha = np.array([alpha]).astype(mat_a.dtype)
        beta  = np.array([beta]).astype(mat_a.dtype)
        self.alpha = AscendArray.clone(alpha)
        self.beta = AscendArray.clone(beta)

    def run(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        a_type = type_map(self.mat_a.dtype)
        b_type = type_map(self.mat_a.dtype)
        c_type = type_map(self.mat_a.dtype)
        # do gemm asyncronize
        ret = acl.blas.gemm_ex(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, 
                                self.m, self.n, self.k, 
                                self.alpha.ascend_data,
                                self.mat_a.ascend_data, self.k, a_type, 
                                self.mat_b.ascend_data, self.n, b_type, 
                                self.beta.ascend_data,
                                self.mat_c.ascend_data, self.n, c_type, 
                                self.high_prec, 
                                self.stream) 
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do blas gemm_ex, return {ret}.")

        # do synchronize stream 
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to synchronize stream in running blas gemm_ex, return {ret}.")
    
    @property
    def out(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        return self.mat_c

    def free_ab(self):
        """ free alpha and beta data memory
        Args:
            None

        Returns:
            None
        """
        if hasattr(self, 'alpha'):
            del self.alpha

        if hasattr(self, 'beta'):
            del self.beta

    def __del__(self):
        if hasattr(self, 'stream'):
            ret = acl.rt.destroy_stream(self.stream)
            assert ret == ACL_SUCCESS, f"destroy stream failed, return {ret}."
        
        # free alpha and beta data memory
        self.free_ab()



class Vmul():
    """ define a Matmul object, release the function .

    Attributes::
        context: the output image bind with an AscendArray object, image.shape(tupe(h, w, c))
        mat_a  : 
        mat_b  : 
        mat_c  :

    Methods:
        __pre_compute : prepare input data
        run           : do compute matmul
        out           : return output result
    """
    def __init__(self, context, mat_a, vec_x, vec_y, alpha=1.0, beta=0.0, highprec=True):
        if not isinstance(context, int):
            raise TypeError(f"Input context expects an int, but got {type(context)}.")

        if not isinstance(mat_a, AscendArray):
            raise TypeError(f"Input mat_a expects an AscendArray, but got {type(mat_a)}.")

        if not isinstance(vec_x, AscendArray):
            raise TypeError(f"Input vec_x expects an AscendArray, but got {type(vec_x)}.")

        if not isinstance(vec_y, AscendArray):
            raise TypeError(f"Input vec_y expects an AscendArray, but got {type(vec_y)}.")

        assert mat_a.format == vec_x.format, f"Input mat_a and vec_x expects same format."

        # assign self value
        self.context = context
        self.stream = create_stream(context)
        self.highprec = 1 if highprec else 0

        # set op model dir
        ret = acl.op.set_model_dir("op_models")
        if ret != ACL_SUCCESS:
            raise ValueError(f"op set model dir failed, return {ret}.")
        
        # calculate m, n, k and trans alpha and beta to np.ndarray
        self.__pre_compute(mat_a, vec_x, vec_y, alpha, beta)

        # do blas gemm_ex and synchronize stream
        self.run()

        # free input data memory
        self.free_in()


    def __pre_compute(self, mat_a, vec_x, vec_y, alpha, beta):
        """ calculate m, n, k and copy alpha/beta to device.
        Args:
            mat_a : (AscendArray) matrix A
            vec_x : (AscendArray) vector x
            vec_y : (AscendArray) vector y
            alpha : (float value)
            beta  : (float value)

        Returns:
            None
        """
        if mat_a.format in [
            PIXEL_FORMAT_YUV_SEMIPLANAR_420,
            PIXEL_FORMAT_YVU_SEMIPLANAR_420
            ]:
            self.m = mat_a.shape[-1]
            self.n = mat_a.shape[0] * 2 // 3
            self.k = mat_b.shape[1] * 2 // 3
        elif mat_a.format in [
            PIXEL_FORMAT_YUV_SEMIPLANAR_422,
            PIXEL_FORMAT_YVU_SEMIPLANAR_422
            ]:
            self.m = mat_a.shape[-1]
            self.n = mat_a.shape[0] * 2 // 3
            self.k = mat_b.shape[1] * 2 // 3
        elif mat_a.format == 'NCHW':
            self.m = mat_a.shape[-1]
            self.n = mat_a.shape[0]
            self.k = mat_b.shape[1]
        elif mat_a.format == 'NHWC':
            self.m = mat_a.shape[-1]
            self.n = mat_a.shape[0]
            self.k = mat_b.shape[1]
        else:
            raise ValueError(f"Input data format not support.")

        alpha = np.array([alpha]).astype(mat_a.dtype)
        beta  = np.array([beta]).astype(mat_a.dtype)
        self.alpha = AscendArray.clone(alpha)
        self.beta = AscendArray.clone(beta)

    def run(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        a_type = type_map(self.mat_a.dtype)
        x_type = type_map(self.vec_x.dtype)
        y_type = type_map(self.vec_y.dtype)
        # do vmul asyncronize
        ret = acl.blas.gemv_ex(ACL_TRANS_N, 
                                self.m, self.n,
                                self.alpha.ascend_data, 
                                self.mat_a.ascend_data, self.k, a_type, 
                                self.vec_x.ascend_data, incx, x_type, 
                                self.beta.ascend_data, 
                                self.vec_y.ascend_data, incy, y_type, 
                                self.high_prec,
                                self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do blas gemv_ex, return {ret}.")

        # do synchronize stream 
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to synchronize stream in running blas gemv_ex, return {ret}.")
    
    @property
    def out(self):
        """ run op.
        Args:
            None

        Returns:
            vec_y
        """
        return self.vec_y

    def _free_ab(self):
        """ free alpha and beta data memory
        Args:
            None

        Returns:
            None
        """
        if hasattr(self, 'alpha'):
            del self.alpha

        if hasattr(self, 'beta'):
            del self.beta

    def __del__(self):
        if hasattr(self, 'stream'):
            ret = acl.rt.destroy_stream(self.stream)
            assert ret == ACL_SUCCESS, f"destroy stream failed, return {ret}."
        
        # free alpha and beta data memory
        self._free_ab()

