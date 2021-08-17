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
from ..resource.mem import *
from ..resource.context import bind_context
from ..common import const

class AscendArray():
    """Define a AscendArray data class like numpy ndarray.

        class private attributes:
        _nbytes (int)    : the bytes of AscendArray's data
        _shape (tuple)   : the shape of this array
        _dtype (int)     : the acl data type of AscendArray
        _flag (int)      : the flag of defined memory malloc on device/host/dvpp
        _mem (instance)  : save the instance of class Memory()
        _data (a pointer): the mem ptr of AscendArray's data

    Attributes:
        ascend_data (int): Get(read) the pointer of data malloced by itself(_data's value), 
            or band it to a new memory.
        shape (tuple): Tuple of array dimensions. (should work with write ascend_data).
        ndim (int): Number of array dimensions.
        dtype (np.dtype): get or write the data type.
        size (int): Number of elements in the array.
        nbytes (int): Total bytes consumed by the elements of the array.
        itemsize (int): Length of one AscendArray element in bytes, which equal to size * dtype(size).
        flag (str): Information about the memory layout of the array.

    .. hint:: 
        - reshape   : Gives a new shape to an array without changing its data.
        - resize    : Return a new array with the specified shape.
        - to_np     : Use this function to copy device data to host and formal like numpy array.
        - to_ascend : trans a ndarray data to AscendArray(i.e. feed this instance with a numpy array data).
        - clone     : new a AscendArray object clone from np.ndarray
        - to        : copy this instance's data to another same shape AscendArray.
        - astype    : Copy of the array, cast to a specified type.
        - transpose : Reverse or permute the axes of an array; returns the modified array.
    """

    def __init__(self, shape, dtype, size=0, buffer=None, format=None, context=None, flag='DEVICE'):
        assert isinstance(shape, tuple), \
            f'Input shape of AscendArray instance expects tuple, but got {type(shape)}'

        try:
            self._dtype = np.dtype(dtype)
        except:
            raise TypeError(
                f'Input dtype expect a numpy.dtype, but got {type(dtype)}')

        self._shape = shape
        self._flag = flag
        self._context = context
        self._format = format

        bind_context(context)

        is_malloc = True
        # calc memory size according to variable calling
        # 1. initial a scalar
        #    Examples:
        #    --------
        #    >>> AscendArray((), dtype=np.int32)
        if shape == ():
            self._nbytes = self._dtype.itemsize

        # 2. initial a array with shape(2, 3):
        #    Examples:
        #    --------
        #    >>> AscendArray((2, 3), dtype=np.float16)
        elif size <= 0 and buffer is None:
            self._nbytes = int(np.prod(shape) * self._dtype.itemsize)

        # 3. initial a array with shape(6,) and 256 bytes
        #    Examples:
        #    --------
        #    >>> AscendArray((6,), dtype=np.float32, size=256)
        elif buffer is None:
            self._nbytes = size
            self._shape = (shape[0], size//(shape[0] * self._dtype.itemsize))

        # 4. initial a array with shape(16,) and binding memory pointer mem_ptr
        #    Examples:
        #    --------
        #    >>> AscendArray((16,), dtype=np.float16, size=256, buffer=mem_ptr)
        else:
            self._nbytes = size
            self._shape = (shape[0], size//(shape[0] * self._dtype.itemsize))
            is_malloc = False

        # bind memory
        if is_malloc:
            self._mem = Memory(self._context, self._nbytes, flag)
            self._data = self._mem.ptr
        else:
            self._data = buffer

    # 1.ascend_data getter and setter
    @property
    def ascend_data(self): 
        return self._data

    @ascend_data.setter
    def ascend_data(self, dev_ptr: int):
        assert isinstance(dev_ptr, int), \
            f'Function dev_ptr args of input expects int type, but got {type(dev_ptr)}'

        if hasattr(self, '_mem'):
            del self._mem
        self._data = dev_ptr

    # 2.memory location getter
    @property
    def flag(self):
        return self._flag

    # 3.context resource getter and setter
    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context):
        """Binding AscendArray with a new context.

        .. warning::
            Only support the memory of AscendArray create by itself.

        Args:
            context : the context to be binded.

        Returns:
            None.
        """
        assert isinstance(context, int), \
            f"Input context expects int value, but got {type(context)}."

        if not hasattr(self, '_mem'):
            raise ValueError(
                f"This AscendArray instance not support to set context.")
        else:
            del self._mem

        bind_context(context)
        self._mem = Memory(context, self._nbytes, self._flag)
        self._data = self._mem.ptr

    # 4.ndim getter
    @property
    def ndim(self):
        return len(self._shape)

    # 5.shape getter and setter
    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        assert isinstance(shape, tuple), \
            f'Input shape expects a tuple, but got {type(shape)}'

        if len(shape) <= 0 or shape[0] == 0:
            raise ValueError(
                'Input shape is empty or format is invalid in calling function shape')

        self._shape = shape

    # 6.dtype getter and setter
    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        try:
            self._dtype = np.dtype(dtype)
        except:
            raise TypeError(
                f'Input dtype expects a numpy.dtype, but got {type(dtype)}')

    # 7.nbytes getter
    @property
    def nbytes(self):
        return self._nbytes

    # 8.itemsize getter
    @property
    def itemsize(self):
        return self._dtype.itemsize

    # 9.data format
    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, format):
        if not isinstance(format, (str, int)):
            raise TypeError(
                f"Input format expects int or string, but got {type(format)}.")

        self._format = format

    @property
    def size(self):
        return np.prod(self._shape)

    def reshape(self, shape):
        """Gives a new shape to an array without changing its data.

        .. Note::
            Only modify the view of AscendArray.
            
        Args:
            shape (tuple[int]): Input new shape to be reshaped. It should be compatible with 
                the original shape.

        Returns:
            [AscendArray] : The original object with new shape.
        
        Typical usage example:
        ```python
        array = np.random.random(36*64).astype('float32').reshape(36, 64)
        ascend_array = ascend.AscendArray.clone(array)
        ascend_array = ascend_array.reshape(64, 36)
        ```
        """
        assert isinstance(shape, tuple), \
            f'Input shape expects tuple type, but got {type(shape)}.'

        assert np.prod(self._shape) == np.prod(shape), \
            f"The given shape({shape})'s elements should same to {self._shape}."

        self._shape = shape
        return self

    def resize(self, shape: tuple):
        """Resize the shape and data of AscendArray.

        .. Note::
            The data arrangement of AscendArray is modified.

        Args:
            shape (tuple): The resized new shape.

        Returns:
            [AscendArray] : The new object with a new shape.
        """
        assert isinstance(shape, tuple), \
            f'Input args of func reshape expects tuple type, but got {type(shape)}.'

        pass

    def to_numpy(self, nbytes=None):
        """Copy the attributes and data of AscendArray to np.ndarry object.

        Args:
            nbytes (int, optional): The data size of this object to be transformed. Defaults to None.

        Returns:
            [ndarray]: A copyed np.ndarray object

        Typical usage example:
        ```python
        array = np.random.random(3264)
        ascend_array = ascend.AscendArray.clone(array)
        data = ascend_array.to_numpy()
        ```
        """
        if self._data is None:
            raise ValueError('Variable self._data is None in calling function to_np, \
                maybe this AscendArray instance parameter is null.')

        if nbytes and nbytes > self._nbytes:
            raise ValueError(
                f"Input nbytes must lower than {self._nbytes}, but got {nbytes}.")

        if self._flag != 'HOST':
            _nbytes = nbytes if nbytes else self._nbytes

            # copy device data to host
            cloned_array = AscendArray(shape=self._shape, dtype=self._dtype, format=self._format,
                                       context=self._context, flag='HOST')
            memcpy_d2h(cloned_array.ascend_data, self._data, _nbytes)
            numpy_ptr = cloned_array.ascend_data
        else:
            numpy_ptr = self._data

        try:
            np_type = const.numpy_dict[self._dtype]
        except KeyError:
            raise ValueError(
                f"Convert AscendArray data to numpy not support this type {self._dtype}.")

        array = acl.util.ptr_to_numpy(numpy_ptr, self._shape, np_type)

        return array.copy()

    @property
    def to_np(self):
        return self.to_numpy()

    def to_ascend(self, array):
        """Copy all the data of array(np.ndarray) to AscendArray.

        Args:
            array (np.ndarray): Input np.ndarray to be copyed.

        Typical usage example:
        ```python
        array = np.ones(shape=(384, 384), dtype='float16')
        ascend_array = AscendArray(shape=(384, 384), nbytes=array.nbytes, dtype=NPY_USHORT)
        ascend_array.to_ascend(array)
        ```
        """
        if self._data is None:
            raise ValueError('instance arg self._data is None in calling function to_ascend, \
                Maybe this AscendArray instance parameter is null.')

        if self._flag == 'HOST':
            raise ValueError(
                f'Method to_ascend only be used with DEVICE or DVPP memory')

        assert isinstance(array, np.ndarray), \
            f'Function to_ascend args of input expects a np.ndarray object, but got {type(array)}'

        assert (array.shape == self._shape) and (array.nbytes == self._nbytes), \
            'Function to_ascend of input expects same shape and nbytes,' \
            f' but actually we got shape:{array.shape}, nbytes:{array.nbytes}.'

        # get the array pointer for copy data to device
        array_ptr = acl.util.numpy_to_ptr(array)

        # do copy
        memcpy_h2d(self._data, array_ptr, self._nbytes)

    def astype(self, dtype):
        """ Cast a tensor from src data type to dst data type. Firstly, we try to use Cast operator 
            to release this function. If it fails, we use numpy astype method. 
        Args:
            dtype (np.dtype): The data type to be transformed.

        Returns:
            [AscendArray]: The new AscendArray data object.

        Typical usage example:
        ```python
        array = np.random.random(64*64).astype('float32').reshape(64, 64)
        ascend_array = ascend.AscendArray.clone(array)
        ascend_array = ascend_array.astype(np.float16)
        ```
        """
        try:
            from ..ops.op import Cast
            return Cast(self, dtype=dtype, context=self.context).data
        except:
            array = self.to_np.astype(dtype)
            return self.clone(array)

    def transpose(self, axes=None):
        """Reverse or permute the axes of an array, and returns the modified array.

        Args:
            axes ([tuple, list], optional): Permute the axes of array. Defaults to None.

        Returns:
            [AscendArray]: A tranposed AscendArray.

        Typical usage example:
        ```python
        array = np.random.random(64*64).astype('float32').reshape(64, 64)
        ascend_array = ascend.AscendArray.clone(array)
        ascend_array = ascend_array.astype(np.float16)
        ```
        """
        if not isinstance(axes, (tuple, list)):
            raise TypeError(
                f"Input axis expects a tuple or list, but got {type(axes)}.")

        try:
            from ..ops.op import Permute
            return Permute(self, axes=axes).data
        except:
            return np.transpose(self.to_np, axes=axes)

    @classmethod
    def clone(cls, array, context=None, format=None, flag="DEVICE"):
        """New an AscendArray object and clone all the attributes of array(np.ndarray) to it.

        Args:
            array (np.ndarray): A np.ndarray data to be cloned
            context (int, optional): The context resource working on. Defaults to None.
            format (data_format, optional): The cloned AscendArray data format, it should be 'NCHW' 
                or 'NHWC' for tensor, or it will be Ascend image format. Defaults to None.
            flag (str, optional): The Ascendarray memory flag, and it same to Ascend.Memory class. 
                Defaults to "DEVICE".

        Raises:
            TypeError: The input array is not the intance of np.ndarray

        Returns:
            [AscendArray]: A cloned AscendArray object.

        Typical usage example:
        ```python
        array = np.random.random(644)
        data = ascend.AscendArray.clone(array)
        ```
        """        
        assert isinstance(array, np.ndarray), \
            f'Input args array expects class np.ndarray object, but got {type(array)}.'

        if context and not isinstance(context, int):
            raise TypeError(
                f"Input context expects int type, but got {type(context)}.")

        bind_context(context)

        # get the array pointer for copy device data to host
        array_ptr = acl.util.numpy_to_ptr(array)

        # new an AscendArray object shape like input array.
        cloned_array = cls(shape=array.shape, dtype=array.dtype,
                           format=format, context=context, flag=flag)

        # do copy
        memcpy_h2d(cloned_array.ascend_data, array_ptr, array.nbytes)

        return cloned_array

    def to(self, ascendarray):
        """Copy this AscendArray data to another ascendarray(AscendArray).

        Args:
            ascendarray (AscendArray): The dst AscendArray to be assigned

        Typical usage example:
        ```python
        array = np.random.random(64*64).astype('float32').reshape(64, 64)
        ascend_array1 = ascend.AscendArray.clone(array)
        ascend_array2 = ascend.AscendArray((128, 32), dtype=np.float32)
        ascend_array1.to(ascend_array2)
        ```
        """
        if self._data is None:
            raise ValueError("Variable self._data is None in calling function 'to', \
                Maybe this AscendArray instance parameter is null.")

        assert isinstance(ascendarray, AscendArray), \
            f"Input args of func 'to' expects a class of AscendArray, but got {type(array)}."

        assert (self._nbytes <= ascendarray.nbytes) and (self._dtype == ascendarray.dtype), \
            "Shape or dtype of the input AscendArray is different from original."

        memcpy_d2d(ascendarray.ascend_data, self._data, self._nbytes)

    def __len__(self):
        """ Number of elements in the array, same to self.size.
        Args:
            None

        Returns:
            number of elements
        """
        return np.prod(self._shape)

    def __getitem__(self, idx):
        """ get AscendArray data using subscript index
        Args:
            idx : an int or slice object

        Returns:
            data of AscendArray
        """
        if self.dtype not in [
            np.dtype('float32'),
            np.dtype('int8'),
            np.dtype('int32'),
            np.dtype('uint8')
        ]:
            raise TypeError("Only dtype in ['float32', 'int32', 'int8', 'uint8'] are support \
                            to using subscript index.")

        if not hasattr(self, '_cloned_array'):
            self._cloned_array = self.to_np
            self._cloned_array.reshape(self._shape)

        if isinstance(idx, int):
            if idx < self.size and idx >= 0:
                return self._cloned_array[idx]
            elif idx > -self.size and idx < 0:
                return self._cloned_array[idx + self.size]
            else:
                raise IndexError(f"index {idx} is out of bounds for axis 0")

        elif isinstance(idx, slice):
            return self._cloned_array[idx]
        else:
            return 'index error'

    def __setitem__(self, index, value):
        """ release to set AscendArray data using subscript index
        Args:
            idx : an int or slice object

        Returns:
            data of AscendArray
        """
        import pdb
        pdb.set_trace()
        if not hasattr(self, '_cloned_array'):
            self._cloned_array = self.to_np

        if isinstance(index, int) and index < self.size:
            self._cloned_array[index] = value
        elif isinstance(index, (list, tuple)):
            for i in index:
                assert i < self.size, f'index out of range.'
            if isinstance(value, (list, tuple)):
                if len(index) == len(value):
                    for i, v in enumerate(index):
                        self._cloned_array[v] = value[i]
                else:
                    raise Exception(
                        'values and index must be of the same length')
            elif isinstance(value, (int, float, str)):
                for i in index:
                    self._cloned_array[i] = value
            else:
                raise Exception('value error')
        else:
            raise Exception('index error')

        # update to device, it has lower performance for always write
        self.to_ascend(self._cloned_array)

    def __repr__(self):
        """ release to represent AscendArray data
        Args:
            None

        Returns:
            repr
        """
        repr = "ascendarray(\n{0}, dtype={1})".format(self.to_np, self.dtype)
        return repr

    def __del__(self):
        if hasattr(self, '_mem'):
            del self._mem
            self._data = None
        elif hasattr(self, '_data') and self._data is not None:
            free(self._data, flag=self._flag)


if __name__ == "__main__":
    import pdb
    from resource.context import Context
    resource = Context({12})
    size = 1382400
    ptr, ret = acl.media.dvpp_malloc(size)
    pdb.set_trace()
    array = AscendArray((1280,), np.dtype('uint8'),
                        size=size, buffer=ptr, flag='DEVICE')

    array = np.ones(shape=(384, 384), dtype='float32')
    ascend_array = AscendArray.clone(array)
    print(ascend_array[2])
    print(ascend_array[:32])
    if isinstance(ascend_array, AscendArray):
        print("clone ndarray success.")
    out = ascend_array.to_np
    print(out)
    del resource
