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

from enum import Enum
from ..common.log import Log
from ..common.const import *
from ..common.path import is_filepath
from ..data.ascendarray import AscendArray
from .venc import Venc
from .frame import Frame


class Status(Enum):
    """An enum that defines decode work status.

    Contains `IDLE`, `READY`, `START`, `RUN`, `END`.
    """
    IDLE = 0
    READY = 1
    RUN = 2
    END = 3

class VideoWriter():
    """Define a VideoWriter class to encode video stream with format h264/265.

    Attributes:
        encode (Venc obj): The DVPP video encoder object.
        fps (int): frame per second
        frameSize (tuple): frame shape to be encoded, like (w,h).
        width (int): Width of the frames in the video stream
        height (int): Height of the frames in the video stream.
        frames (int): Number of frames in the video file.
        format (int): video stream format
        fps (float): Frame rate.
        bitrate (int): (read-only) Video bitrate in kbits/s
        pix_fmt (int): the encode frame's pixel format, only support yuv420 NV21/12

    Methods:
        - write   : Writes the next video frame
        - set     : Sets a property in the VideoWriter
        - get     : Returns the specified VideoWriter property
        - release : Closes and release video writer's resource

    """
    def __init__(self, context, filename, fps, frameSize, is_color=False):
        if not isinstance(context, int):
            raise TypeError(f"VideoWriter input context expects an int, bug got {type(context)}.")

        if not is_filepath(filename):
            raise TypeError(f"VideoWriter input filename expects an str or Path, bug got {type(filename)}.")

        if not isinstance(fps, int):
            raise TypeError(f"VideoWriter input fps expects an int, bug got {type(fps)}.")

        if not isinstance(frameSize, tuple) or len(frameSize) != 2:
            raise TypeError(f"VideoWriter input frameSize expects a 2-elements tuple, bug got {type(fps)}.")

        self._context = context

        # check input param ok
        self._status = Status.IDLE

        # initial encoder
        self.encode = Venc(context, filename, frameSize[0], frameSize[1])

        # configure fps
        self.encode.fps = fps

        # intial ok, and step to READY status
        self._status = Status.READY

    @property
    def width(self):
        return self.encode.width

    @width.setter
    def width(self, width):
        if self._status == Status.READY:
            self.encode.width = width
            return True
        else:
            return False

    @property
    def height(self):
        return self.encode.height

    @height.setter
    def height(self, height):
        if self._status == Status.READY:
            self.encode.height = height
            return True
        else:
            return False

    @property
    def fps(self):
        return self.encode.fps

    @fps.setter
    def fps(self, fps):
        if self._status == Status.READY:
            self.encode.fps = fps
            return True
        else:
            return False

    @property
    def pix_fmt(self):
        return self.encode.pic_format

    @pix_fmt.setter
    def pix_fmt(self, pix_fmt):
        if self._status == Status.READY:
            self.encode.pic_format = pix_fmt
            return True
        else:
            return False


    def write(self, image):
        """The function/method writes the specified image to video file. It must have the same size as has
            been specified when opening the video writer.
        Args:
            image (AscendArray): The written frame. In general, color images are expected in BGR format.

        """
        if self._status in [Status.READY, Status.RUN] and image:
            if not isinstance(image, AscendArray):
                self._status = Status.END
                raise TypeError(f"Input image expects an AscendArray, but got {type(image)}.")

            frame = Frame(image)
            self.encode.process(frame)
            self._status = Status.RUN
            Log(INFO, 'write one frame to VideoWriter.')

        else:
            Log(WARNING, f'write image in status {self._status}.')

    def set(self, attr, value):
        """ set(attr, value) -> retval
            Sets a property in the VideoWriter
        Args:
            attr (str): Property from VideoWriter Properties (eg. 'width', 'fps', ...)
            value (int): Value of the property

        Returns:
            bool : `True` if the property is supported by the backend used by the VideoWriter instance.
        """
        if self._status == Status.READY:
            if attr == 'width': 
                self.encode.width = value
            elif attr == 'height': 
                self.encode.height = value
            elif attr == 'fps': 
                self.encode.fps = value
            elif attr == 'enc_type': 
                self.encode.encoder_type = value
            elif attr == 'pix_fmt': 
                self.encode.pic_format = value
            elif attr == 'bit_rate': 
                self.encode.bit_rate = value
            elif attr == 'key_frame': 
                self.encode.frame_itval = value
            elif attr == 'rc_mode': 
                self.encode.rc_mode = value
            else:
                Log(ERROR, f'attr {attr} is not support in VideoWriter.')
                return False
            return True
        else:
            Log(ERROR, f'Set attr {attr} in status {self._status}.')
            raise ValueError(f"Set attr {attr} in status {self._status}.")

    def get(self, attr):
        """ get(attr) -> retval
            Returns the specified VideoWriter property
        Args:
            attr (str): Property from VideoWriter Properties (eg. 'width', 'fps', ...)
            
        Returns:
            [int]: Value for the specified property. Value 0 is returned when querying a 
                property that is not supported by the backend used by the VideoWriter instance.
        """
        if not isinstance(attr, str):
            raise TypeError(f"Input attr expects a string, but got {type(attr)}.")
        
        attr_dict = {
            'width'       : self.encode.width,
            'height'      : self.encode.height,
            'fps'         : self.encode.fps,
            'enc_type'    : self.encode.encoder_type,
            'pix_fmt'     : self.encode.pic_format,
            'bit_rate'    : self.encode.bit_rate,
            'key_frame'   : self.encode.frame_itval,
            'rc_mode'     : self.encode.rc_mode
        }

        try:
            return attr_dict[attr]
        except KeyError:
            return 

    def release(self):
        """Closes video file or writer and release resource.
        """
        if self._status == Status.RUN:
            frame = Frame(None, is_last=True)
            self.encode.process(frame)
            self._status = Status.END

            # do finish job and stop encoder thread
            self.encode.finish()

            Log(INFO, f"write last frame to VideoWriter.")

        if hasattr(self, 'encode'):
            del self.encode
        
        self._status = Status.END

    def __del__(self):
        if hasattr(self, 'encode'):
            del self.encode
        

