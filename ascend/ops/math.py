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

from ..common.const import *
from ..resource.context import create_stream
from ..data.ascendarray import AscendArray
from .util import TensorDesc, OpSet

# Permute
class FFT():
    """ define a FFT operator object to do Fast Fouries Transform. 

    Attributes:
        input  : input tensor (AscendArray)
        perm   : Permutes the dimensions.
        context: input context, optional
        stream : input stream, optional

    Methods:
        run    : do permute
        out    : return output result
    """
    def __init__(self, input, context=None, stream=None):
        if not isinstance(input, AscendArray):
            raise TypeError(f"Input tensor expects a AscendArray, but got {type(input)}.")

        if context and not isinstance(context, int):
            raise TypeError(f"Input context expects an int, but got {type(context)}.")

        if stream and not isinstance(stream, int):
            raise TypeError(f"Input stream expects an int, but got {type(stream)}.")

        # assign self value
        self.context = context
        self.stream = stream if stream else create_stream(context)
        self.created = stream is None

        OpSet()
        self.__pre_set()

        # create output array to save result
        self.output = AscendArray(input.shape, dtype=input.dtype, format='ND')
        self.tensor_in = TensorDesc(input)
        self.tensor_out = TensorDesc(self.output)

        # do transpose
        self.run()


    def __pre_set(self):
        """ set op name and attribute.
        Args:
            None

        Returns:
            None
        """
        self.op_name = "Permute"
        self.op_attr = acl.op.create_attr()
        ret = acl.op.set_attr_list_int(self.op_attr, 'order', np.array([0, 3, 2, 1], dtype=np.int32))
        if ret != ACL_SUCCESS:
            raise ValueError(f"Set attr 'order' failed, return {ret}.")


    def run(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        # do op excute
        ret = acl.op.execute(self.op_name,
                             [self.tensor_in.desc],
                             [self.tensor_in.buff],
                             [self.tensor_out.desc],
                             [self.tensor_out.buff],
                             self.op_attr,
                             self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to excute op {self.op_name}, return {ret}.")

        # do synchronize stream 
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to synchronize stream in excute op, return {ret}.")
    
    @property
    def data(self):
        return self.output

    def __del__(self):
        if hasattr(self, 'output'):
            del self.output
        if hasattr(self, 'tensor_out'):
            del self.tensor_out
        if hasattr(self, 'tensor_in'):
            del self.tensor_in

        if self.created:
            ret = acl.rt.destroy_stream(self.stream)
            assert ret == ACL_SUCCESS, f"destroy stream failed, return {ret}."


# Permute
class IFFT():
    """ define a Permute operator object to permute the dimensions. 

    Attributes:
        input  : input tensor (AscendArray)
        perm   : Permutes the dimensions.
        context: input context, optional
        stream : input stream, optional

    Methods:
        run    : do permute
        out    : return output result
    """
    def __init__(self, input, context=None, stream=None):
        if not isinstance(input, AscendArray):
            raise TypeError(f"Input tensor expects a AscendArray, but got {type(input)}.")

        if context and not isinstance(context, int):
            raise TypeError(f"Input context expects an int, but got {type(context)}.")

        if stream and not isinstance(stream, int):
            raise TypeError(f"Input stream expects an int, but got {type(stream)}.")

        # assign self value
        self.context = context
        self.stream = stream if stream else create_stream(context)
        self.created = stream is None

        OpSet()
        self.__pre_set()

        # create output array to save result
        self.output = AscendArray(input.shape, dtype=input.dtype, format='ND')
        self.tensor_in = TensorDesc(input)
        self.tensor_out = TensorDesc(self.output)

        # do transpose
        self.run()


    def __pre_set(self):
        """ set op name and attribute.
        Args:
            None

        Returns:
            None
        """
        self.op_name = "Permute"
        self.op_attr = acl.op.create_attr()
        ret = acl.op.set_attr_list_int(self.op_attr, 'order', np.array([0, 3, 2, 1], dtype=np.int32))
        if ret != ACL_SUCCESS:
            raise ValueError(f"Set attr 'order' failed, return {ret}.")


    def run(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        # do op excute
        ret = acl.op.execute(self.op_name,
                             [self.tensor_in.desc],
                             [self.tensor_in.buff],
                             [self.tensor_out.desc],
                             [self.tensor_out.buff],
                             self.op_attr,
                             self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to excute op {self.op_name}, return {ret}.")

        # do synchronize stream 
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to synchronize stream in excute op, return {ret}.")
    
    @property
    def data(self):
        return self.output

    def __del__(self):
        if hasattr(self, 'output'):
            del self.output
        if hasattr(self, 'tensor_out'):
            del self.tensor_out
        if hasattr(self, 'tensor_in'):
            del self.tensor_in

        if self.created:
            ret = acl.rt.destroy_stream(self.stream)
            assert ret == ACL_SUCCESS, f"destroy stream failed, return {ret}."
