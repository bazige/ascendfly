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


class Cast():
    """ define a Cast operator object to release dtype convert. support translate types:
        float -> float16, float16 -> float, uint8 -> float16, float16 -> uint8
        uint8 -> float32, float32 -> uint8, float16 -> int32, int32 -> float16  

    Args:
        input  : input tensor (AscendArray)
        dtype  : the converted data type of input.
        context: input context, optional
        stream : input stream, optional

    function:
        run             : do compute matmul
        out             : return output result
    """
    def __init__(self, input, dtype=np.dtype('float16'), context=None, stream=None):
        if not isinstance(input, AscendArray):
            raise TypeError(f"Input tensor expects a AscendArray, but got {type(input)}.")

        if context and not isinstance(context, int):
            raise TypeError(f"Input context expects an int, but got {type(context)}.")

        if stream and not isinstance(stream, int):
            raise TypeError(f"Input stream expects an int, but got {type(stream)}.")

        # assign self value
        self.input = input
        self.context = context
        self.stream = stream if stream else create_stream(context)
        self.created = stream is None

        OpSet()
         # create output array to save result
        self.output = AscendArray(input.shape, dtype=dtype, format='ND')
    
        self.tensor_in = TensorDesc(input)
        self.tensor_out = TensorDesc(self.output)

        # do cast operator
        self.run()
        
    def run(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        # do op cast
        ret = acl.op.cast(self.tensor_in.desc,
                          self.tensor_in.buff,
                          self.tensor_out.desc,
                          self.tensor_out.buff,
                          0,
                          self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do op cast, return {ret}.")

        # do synchronize stream 
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to synchronize stream in running blas gemm_ex, return {ret}.")
    
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


class ArgMax():
    """ define a ArgMax operator.

    Args:
        input  : input tensor (AscendArray)
        size   : output data size
        context: input context, optinal
        stream : input stream, optinal

    function:
        run    : do argmax
        out    : return output result
    """
    def __init__(self, input, axis=0, context=None, stream=None):
        if not isinstance(input, AscendArray):
            raise TypeError(f"Input tensor expects a AscendArray, but got {type(input)}.")

        if context and not isinstance(context, int):
            raise TypeError(f"Input context expects an int, but got {type(context)}.")

        if stream and not isinstance(stream, int):
            raise TypeError(f"Input stream expects an int, but got {type(stream)}.")

        if axis >= input.ndim:
            raise ValueError(f"Input axis should in range [0, {input.ndim}).")

        # assign self value
        self.input = input
        self.context = context
        self.stream = stream if stream else create_stream(context)
        self.created = stream is None

        # set op model dir
        OpSet()
        self.__pre_set()

        # create output array to save result
        shape = input.shape[:axis] + input.shape[axis + 1:]
        self._dim = AscendArray.clone(np.array(axis, dtype=np.int32))
        self._out = AscendArray(shape, dtype=np.int32)

        self.tensor_in = TensorDesc(input)
        self.tensor_dim = TensorDesc(self._dim)
        self.tensor_out = TensorDesc(self._out)

        # do cast operator
        self.run()

    def __pre_set(self):
        """ set op name and attribute.
        Args:
            None

        Returns:
            None
        """
        self.op_name = "ArgMaxV2"
        self.op_attr = acl.op.create_attr()

    def run(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        # do op excute
        ret = acl.op.execute(self.op_name,
                             [self.tensor_in.desc, self.tensor_dim.desc],
                             [self.tensor_in.buff, self.tensor_dim.buff],
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
        return self._out

    def __del__(self):
        if hasattr(self, '_out'):
            del self._out
        if hasattr(self, 'tensor_out'):
            del self.tensor_out
        if hasattr(self, 'tensor_in'):
            del self.tensor_in

        if self.created:
            ret = acl.rt.destroy_stream(self.stream)
            assert ret == ACL_SUCCESS, f"destroy stream failed, return {ret}."
        

#Transpose
class Transpose():
    """ define a Transpose operator object to release Transpose. Permutes the dimensions according to perm. 
        The returned tensor's dimension i will correspond to the input dimension perm[i] 

    Args:
        input  : input tensor (AscendArray)
        perm   : Permutes the dimensions.
        context: input context, optional
        stream : input stream, optional

    Methods:
        run    : do permute
        out    : return output result
    """
    def __init__(self, input, perm=[0, 1, 2, 3], context=None, stream=None):
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
        _perm = AscendArray.clone(np.array(perm, dtype='int32'))
        self.output = AscendArray(input.shape, dtype=input.dtype, format='ND')
        self.tensor_in1 = TensorDesc(input)
        self.tensor_in2 = TensorDesc(_perm)
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
        self.op_name = "Transpose"
        self.op_attr = acl.op.create_attr()


    def run(self):
        """ run op.
        Args:
            None

        Returns:
            None
        """
        # do op excute
        ret = acl.op.execute(self.op_name,
                             [self.tensor_in1.desc, self.tensor_in2.desc],
                             [self.tensor_in1.buff, self.tensor_in2.buff],
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
class Permute():
    """ define a Permute operator object to permute the dimensions. 

    Attributes::
        input  : input tensor (AscendArray)
        axes   : Permutes the dimensions.
        context: input context, optional
        stream : input stream, optional

    Methods:
        run    : do permute
        out    : return output result
    """
    def __init__(self, input, axes=(0, 2, 3, 1), context=None, stream=None):
        if not isinstance(input, AscendArray):
            raise TypeError(f"Input tensor expects a AscendArray, but got {type(input)}.")
        
        if not isinstance(axes, (tuple, list)):
            raise TypeError(f"Input axes expects a tuple or list, but got {type(axes)}.")

        if context and not isinstance(context, int):
            raise TypeError(f"Input context expects an int, but got {type(context)}.")

        if stream and not isinstance(stream, int):
            raise TypeError(f"Input stream expects an int, but got {type(stream)}.")

        if tuple(axes) not in [(0, 2, 3, 1), (0, 3, 1, 2)]:
            raise ValueError(f"Input axis only support (0, 2, 3, 1) or (0, 2, 3, 1).")
            
        # assign self value
        self._axes = axes
        self.context = context
        self.stream = stream if stream else create_stream(context)
        self.created = stream is None

        OpSet()
        self.__pre_set()

        # create output array to save result
        out_shape = tuple([input.shape[i] for i in axes])
        self.output = AscendArray(out_shape, dtype=input.dtype, format='ND')
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
        ret = acl.op.set_attr_list_int(self.op_attr, 'order', np.array(self._axes, dtype=np.int64))
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

        if hasattr(self, 'created') and self.created:
            ret = acl.rt.destroy_stream(self.stream)
            assert ret == ACL_SUCCESS, f"destroy stream failed, return {ret}."
