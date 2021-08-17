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

import acl
import numpy as np
import os.path as osp

from ..common.const import *
from ..data.ascendarray import AscendArray

dtyte_trans = {
    np.dtype('float32'):ACL_FLOAT  ,
    np.dtype('float16'):ACL_FLOAT16,
    np.dtype('int8')   :ACL_INT8   ,
    np.dtype('int32')  :ACL_INT32  ,
    np.dtype('uint8')  :ACL_UINT8  ,
    np.dtype('int16')  :ACL_INT16  ,
    np.dtype('uint16') :ACL_UINT16 ,
    np.dtype('uint32') :ACL_UINT32 ,
    np.dtype('int64')  :ACL_INT64  ,
    np.dtype('uint64') :ACL_UINT64 ,
    np.dtype('float64'):ACL_DOUBLE ,
    np.dtype('bool')   :ACL_BOOL   
}


class TensorDesc():
    """ define a TensorDesc object, and bind with AscendArray mem.

    Args:
        tensor  : input tensor
        dtype   : reset tensor data with a new dtype
    """
    def __init__(self, tensor, dtype=None):
        if not isinstance(tensor, AscendArray):
            raise TypeError(f"Input tensor expects an AscendArray, but got {type(tensor)}.")

        dtype = tensor.dtype if dtype is None else dtype
        try:
            data_type = dtyte_trans[dtype]
        except KeyError:
            raise ValueError(f"Input dtype {dtype} is not support.")

        if tensor.ndim == 0:
            list_dims = [1]
        else:
            list_dims = list(tensor.shape)

        self.desc = acl.create_tensor_desc(data_type, list_dims, ACL_FORMAT_ND)

        # bind data buffer with AscendArray
        self.buff  = acl.create_data_buffer(tensor.ascend_data, tensor.nbytes)

    def __del__(self):
        if hasattr(self, 'buff'):
            ret = acl.destroy_data_buffer(self.buff)
            assert ret == ACL_SUCCESS, f"destroy data buffer failed, return {ret}."
        
        acl.destroy_tensor_desc(self.desc)
        
class OpSet():
    """ define a single intance object, to set op model one time.

    Args:
        tensor  : input tensor
        dtype   : reset tensor data with a new dtype
    """
    def __init__(self):
        if type(self)._first:
            dir_name = osp.abspath(osp.dirname(__file__))
            ret = acl.op.set_model_dir(dir_name + "/om")
            if ret != ACL_SUCCESS:
                raise ValueError(f"op set model dir failed, return {ret}.")

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_inst'):   
            cls._inst = super(OpSet, cls).__new__(cls)
            cls._first = True
        else:
            cls._first = False
        return cls._inst

