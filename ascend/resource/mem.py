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

from ..common import const
from ..resource.context import bind_context

def malloc(size, flag):
    """Malloc memory according to flag.
    Args:
        size (int): The memory size to malloc
        flag (str): The memory's location, one of ['DVPP', 'DEVICE', 'HOST']

    Returns:
        int: The malloc memory's pointer.
    """
    if flag == 'DVPP':
        ptr, ret = acl.media.dvpp_malloc(size)
    elif flag == 'DEVICE':
        ptr, ret = acl.rt.malloc(size, const.ACL_MEM_MALLOC_HUGE_FIRST)
    elif flag == 'HOST':
        ptr, ret = acl.rt.malloc_host(size)

    if ret != const.ACL_SUCCESS:
        ptr = None
        raise ValueError(f"Failed to alloc {flag} memory, return value:{ret}.")
        
    return ptr

def free(ptr, flag):
    """Free memoy at point ptr.
    Args:
        ptr (int): The pointer of memory to be released
        flag (str): The memory's loaction
    """
    if flag == 'DVPP':
        ret = acl.media.dvpp_free(ptr)
    elif flag == 'DEVICE':
        ret = acl.rt.free(ptr)
    elif flag == 'HOST':
        ret = acl.rt.free_host(ptr)
        
    if ret != const.ACL_SUCCESS:
        raise ValueError(f"Failed to free {flag} memory at {ptr}, return value {ret}.")

def mem_copy(dst_ptr, src_ptr, size, method):
    """Copy memory from source to destination.
    Args:
        dst_ptr (int): The pointer of dst memory
        src_ptr (int): The pointer of src memory
        size (int): Memory size to be copyed
        method (str): Copy method, one of ['host_to_device', 'device_to_host', 'device_to_device',
            'host_to_host']

    Returns:
        None.
    """
    if not isinstance(dst_ptr, int):
        raise TypeError(f"Input dst_ptr expects an int, but got {type(dst_ptr)}.")

    if not isinstance(src_ptr, int):
        raise TypeError(f"Input dst_ptr expects an int, but got {type(dst_ptr)}.")

    if not isinstance(size, int):
        raise TypeError(f"Input dst_ptr expects an int, but got {type(dst_ptr)}.")

    assert size > 0, f"Input size expects a positive value, but got {size}."

    if method == 'host_to_device':
        ret = acl.rt.memcpy(dst_ptr, size, src_ptr, size,  const.ACL_MEMCPY_HOST_TO_DEVICE)
    elif method == 'device_to_host':
        ret = acl.rt.memcpy(dst_ptr, size, src_ptr, size,  const.ACL_MEMCPY_DEVICE_TO_HOST)
    elif method == 'device_to_device':
        ret = acl.rt.memcpy(dst_ptr, size, src_ptr, size,  const.ACL_MEMCPY_DEVICE_TO_DEVICE)
    elif method == 'host_to_host':
        ret = acl.rt.memcpy(dst_ptr, size, src_ptr, size,  const.ACL_MEMCPY_HOST_TO_HOST)
    else:
        raise ValueError(f"Input memory copy method {method} is not support," \
            "only support ['host_to_host', 'device_to_device', 'host_to_device', 'device_to_host'].")
    
    if ret != const.ACL_SUCCESS:
        raise ValueError(f"Copy src memory {src_ptr} to dst {dst_ptr} with {method} failed, return {ret}.")

def memcpy_h2d(dst_ptr, src_ptr, size):
    """Same to function `mem_copy` with method `'host_to_device'`
    """    
    return mem_copy(dst_ptr, src_ptr, size, 'host_to_device')

def memcpy_d2h(dst_ptr, src_ptr, size):
    """Same to function `mem_copy` with method `'device_to_host'`
    """   
    return mem_copy(dst_ptr, src_ptr, size, 'device_to_host')

def memcpy_d2d(dst_ptr, src_ptr, size):
    """Same to function `mem_copy` with method `'device_to_device'`
    """   
    return mem_copy(dst_ptr, src_ptr, size, 'device_to_device')

def memcpy_h2h(dst_ptr, src_ptr, size):
    """Same to function `mem_copy` with method `'host_to_host'`
    """   
    return mem_copy(dst_ptr, src_ptr, size, 'host_to_host')



class Memory():
    """Define a Memory class to manage memory of device host and dvpp.

    Attributes:
        ptr (a pointer): the memmory ptr
        size (int value): the memory size
        flag (string): one of ['DVPP', 'DEVICE', 'HOST']

    Typical usage example:
    ```python
    mem = ascend.Memory(context, 32768, flag='DVPP')
    print(mem.ptr)
    ```
    """
    def __init__(self, context, size, flag="DEVICE"):
        if not isinstance(size, int):
            raise TypeError('Memory instance input praram size expects an int value. '
                                f'But received {type(size)}')
        if size <= 0:
            raise ValueError('Memory instance input praram size should larger than 0. '
                                f'But received {size}')

        if flag not in ['DVPP', 'DEVICE', 'HOST']:
            raise TypeError("Memory instance input flag expects one of ['DVPP', 'DEVICE', 'HOST'].")

        # set context
        bind_context(context)

        self.size = size
        self.flag = flag
        self.ptr = malloc(size, flag)

    @classmethod
    def reset(self, ptr, size, value):
        """Initial memory with a const value.

        Args:
            ptr (int): Input memory pointer.
            size (int): Input memory size to be reset.
            value (int): The memory value to reset.

        Typical usage example:
        ```python
        pad_val = 0
        ascend.Memory.reset(dst_img.ascend_data, dst_img.nbytes, pad_val)
        ```
        """
        if not isinstance(ptr, int):
            raise TypeError("Input praram 'ptr' of func reset expects an int value. "
                                f'But received {type(ptr)}')
        if not isinstance(size, int):
            raise TypeError("Input praram 'size' of func reset expects an int value. "
                                f'But received {type(size)}')
        if not isinstance(value, int):
            raise TypeError("Input praram 'value' of func reset expects an int value. "
                                f'But received {type(value)}')

        ret = acl.rt.memset(ptr, size, value, size)
        if ret != const.ACL_SUCCESS:
            raise ValueError(f"Failed to memset at pointer {ptr}, return value:{ret}.")
        
    def mem_info(self, attr):
        """Get the memory info accoring to attribute att.
        Args:
            attr (int): Input attribute info to get.

        Returns:
            tuple(free, total): The free memory and total memory of device
        
        Typical usage example:
        ```python
        mem = ascend.Memory(context, 32768)
        mem_info = mem.mem_info(0)
        ```
        """
        if not isinstance(attr, int):
            raise TypeError(f"Input param expects an int, but got {type(attr)}.")

        free, total, ret = acl.rt.get_mem_info(attr)
        if ret != const.ACL_SUCCESS:
            raise ValueError(f"Get mem info failed, return {ret}.")

        return free, total

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            free(self.ptr, self.flag)



