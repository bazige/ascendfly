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
from av.packet import Packet
from ascend.data.ascendarray import AscendArray
from ..resource.mem import Memory, memcpy_h2d

class Frame():
    """Define a Frame object to manage push packet to video_decoder or image(AscendArray)
       to video_encoder.
 
    Attributes:
        data (int): the Frame data pointer
        width (int): width info of Frame
        height (int): height info of Frame
        size (int): buffer size of this frame
        is_last (bool): last frame flag
        frame_id (iint): id number of this Frame

    """
    def __init__(self, packet, shape=None, is_last=False, frame_id=0, context=None):
        self._packet = packet          
        self._is_last = is_last
        self._frame_id = frame_id
        self._shape = shape

        # check input para
        self.__check_para()

        if isinstance(packet, AscendArray):
            self._shape = packet.shape[::-1]

        # malloc device memory, and copy host data to it
        if not is_last and isinstance(packet, Packet):
            self._mem = Memory(context, packet.buffer_size, flag="DVPP")

            memcpy_h2d(self._mem.ptr, packet.buffer_ptr, packet.buffer_size)

    def __check_para(self):
        """ check input parameters of instance Frame.
        Args:
            context : the device context resource.
            string  : input stream.
            channel : the channel id of Vdec.
            en_type : the encode type of stream.

        Returns:
            None.
        """
        if self._is_last:
            return

        if not isinstance(self._packet, (Packet, AscendArray)):
            raise TypeError(f"Input packet expects a av.packet.Packet or AscendArray, but got {type(self._packet)}.")

        if self._shape and not isinstance(self._shape, tuple):
            raise TypeError(f"Input shape expects a tupe, but got {type(shape)}.")

        if isinstance(self._packet, Packet) and self._packet.buffer_size <= 0:
            raise ValueError(f"Input packet size expects a positive value, but actually {self._packet.buffer_size}.")
        elif isinstance(self._packet, AscendArray) and self._packet.nbytes <= 0:
            raise ValueError(f"Input packet size expects a positive value, but actually {self._packet.nbytes}.")

    @property
    def data(self):
        if hasattr(self, '_mem'):
            return self._mem.ptr
        else:
            return self._packet.ascend_data


    @property
    def width(self):
        return self._shape[0]

    @property
    def height(self):
        return self._shape[1]

    @property
    def size(self):
        if isinstance(self._packet, Packet):
            return self._packet.buffer_size
        else:
            return self._packet.nbytes

    @property
    def is_last(self):
        return self._is_last

    @property
    def frame_id(self):
        return self._frame_id

    def __del__(self):
        if hasattr(self, "_mem") and self._mem.ptr:
            del self._mem


