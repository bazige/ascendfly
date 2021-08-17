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

import av
import av.datasets
import numpy as np
from enum import Enum
from ..common.log import Log
from ..common.const import *
from .vdec import Vdec
from .frame import Frame
from ..resource.context import bind_context

class Status(Enum):
    """An enum that defines decode work status.

    Contains `IDLE`, `READY`, `START`, `RUN`, `END`.
    """
    IDLE = 0
    READY = 1
    START = 2
    RUN = 3
    END = 4


class VideoCapture():
    """Define a VideoCaputure class to get the stream and parser packet. It also include method to push 
       packets to ascend dvpp video decoder, and the decoder only support annex-b h264 video format or
       rtsp ip camera. 
    
    .. warning::
       Input an IP video stream for video capturing, and stream_path expect an absolute path for video dir. 

    Attributes:
        context (int): context resource video capture working on
        container (av.Container obj): The pyav container object.
        stream (container.streams obj): The pyav Container streams object.
        packets (av.Packet obj): The pyav Packet object.
        decoder (Vdec obj): The ascend DVPP decoder object.
        width (int): frame's width
        height (int): frame's height
        coded_height (int): video coded height
        codel_width (int): video coded width
        display_aspect_ratio (Fraction obj): get the video stream display aspect ratio.
        format (av.VideoFormat obj): video stream format.
        framerate (float): video stream framerate
        fps (float): video stream framerate, same to framerate
        has_b_frames (bool): bool value, indicate stream has b frames
        pix_fmt (int): the decoded frame's pixel format
        frames (int): frame number of video stream


    Methods:
        - skip_frame : Signal that we only want to look at frames, only work in ffmpeg
        - is_open    : Returns true if video capturing has been initialized already
        - read       : Grabs, decodes and returns the next video frame
        - set        : Sets a property in the VideoCapture
        - get        : Returns the specified VideoCapture property
        - release    : Closes and release video capture's resource

    """
    def __init__(self, context, stream_path, channel=None):
        if not isinstance(stream_path, str):
            raise TypeError(f"Input stream_path expects an string in initial VideoCapture, \
                 but got {type(stream_path)}.")

        if not isinstance(context, int):
            raise TypeError(f"VideoCaputre input context expects an int type, bug got {type(context)}.")

        self.context = context
        self._status = Status.IDLE

        # open video container
        self.container = av.open(av.datasets.curated(stream_path))

        # only use video stream
        self.stream = self.container.streams.video[0]
        
        # check video format is support or not, and get en_type of video stream.
        en_type = self.__check_stream_format(self.stream.codec_context)
        
        # push stream.
        self.stream.thread_type = 'AUTO'
        self.packets = self.container.demux()

        # set context
        bind_context(context)

        pix_fmt = pix_fmt_map.get(self.stream.pix_fmt, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        chan_id = channel if channel else self.stream.id
        self.decoder = Vdec(context, en_type=en_type, pix_fmt=pix_fmt, channel=chan_id)

        # set initial working status
        self._frame = 0
        self._status = Status.READY

    def __check_stream_format(self, codec_context):
        """ check the video stream format is Annex-b or not, and Annex-b format h264 extradata is start
            with 0x000001 or 0x00000001
        Args:
            stream : input stream

        Returns:
            None
        """
        extradata = np.frombuffer(codec_context.extradata, np.ubyte)
        if (extradata[:3] == [0, 0, 1]).all():
            profile_id = extradata[4]
        elif (extradata[:4] == [0, 0, 0, 1]).all():
            profile_id = extradata[5]
        else:
            raise ValueError(f"Input stream {stream} is not annex-b h264.")

        if codec_context.name not in ['h264', 'h265']:
            raise ValueError(f"unsupport this video stream codec type {codec_context.name}.")

        try:
            en_type = en_type_map[profile_id]
        except KeyError:
            raise ValueError(f"Input stream format is not one of {en_type_map}.")

        return en_type


    @property
    def width(self):
        return self.stream.width

    @property
    def height(self):
        return self.stream.height

    @property
    def coded_height(self):
        return self.stream.coded_height

    @property
    def coded_width(self):
        return self.stream.coded_width

    @property
    def display_aspect_ratio(self):
        """
        .. note:: 
            Get the video stream display aspect ratio, and it returns an fraction object 
            like Fraction(16, 9).
        """
        return self.stream.display_aspect_ratio

    @property
    def format(self):
        """
        .. note:: 
            Get the video stream format info like <av.VideoFormat yuv420p, 3840x2160>. It 
            return an av.VideoFormat object.
        """
        return self.stream.format

    @property
    def framerate(self):
        return float(self.stream.framerate)

    @property
    def fps(self):
        return self.framerate

    @property
    def has_b_frames(self):
        return self.stream.has_b_frames

    @property
    def pix_fmt(self):
        return self.stream.pix_fmt

    @property
    def frames(self):
        return self.stream.frames

    def skip_frame(self, skip_type):
        """Signal that we only want to look at frames, only work in ffmpeg.
        Args:
            skip_type (int): A class of av.codec.context.SkipType

        ```python
        -------------------------------------------------------------------------------
        SkipType Name | Flag Value | Meaning in FFmpeg                                 
        --------------+------------+---------------------------------------------------
        NONE          | 0x-10      | Discard nothing                                   
        DEFAULT       | 0x0        | Discard useless packets like 0 size packets in AVI
        NONREF        | 0x8        | Discard all non reference                         
        BIDIR         | 0x10       | Discard all bidirectional frames                  
        NONINTRA      | 0x18       | Discard all non intra frames                      
        NONKEY        | 0x20       | Discard all frames except keyframes               
        ALL           | 0x30       | Discard all                                       
        -------------------------------------------------------------------------------
        ```
        """
        if not isinstance(skip_type, str):
            raise TypeError(f"Input skip_type expect a string, but got {type(skip_type)}.")

        if skip_type in ['NONE', 'DEFAULT', 'NONREF', 'BIDIR', 'NONINTRA', 'NONKEY', 'ALL']:
            self.stream.codec_context.skip_frame = skip_type
        else:
            Log(WARNING, 'skip_frame set value failed in Video Capture.')


    def is_open(self):
        """Open the video capture and ready to decode.

        ```python
        The decode working on 5 status: IDLE, READY/START, RUN, END
        =================================================================================
        Status IDLE  | the idle status of decode, and the resource is uninitial, and
                     | nothing can be used in instance. 
        ---------------------------------------------------------------------------------
        Status READY | class VideoCapture and Vdec is initialized, and waiting for start. 
        ---------------------------------------------------------------------------------
        Status START | this status start to open vdec and the dequeue is null, so it will
                     | push packet until dequeue has decoded-image data. 
        ---------------------------------------------------------------------------------
        Status RUN   | run video decoder and pull stream packet until packet is null. 
        ---------------------------------------------------------------------------------
        Status END   | end of push packet to the vdec, and pop image data until the  
                     | dequeue is null. 
        =================================================================================
        ```
        Args:
            None

        Returns:
            bool : True for VideoCapture is ready or False.
        """
        if self._status == Status.START or self._status == Status.READY:
            # get one packet data
            packet = next(self.packets)

            # if packet is null, jump to status IDLE
            if packet.buffer_size <= 0:
                self._status == Status.IDLE
                return False

            # construct a frame
            self._frame = self._frame + 1
            shape = (self.stream.width, self.stream.height)
            frame = Frame(packet, shape, frame_id=self._frame, context=self.context)

            # do video decode
            self.decoder.process(frame)

            # if dequeue has image data, it jump to status RUN
            if not self.decoder.image.empty():
                self._status = Status.RUN
            return True

        elif self._status == Status.RUN:
            # get one packet data
            packet = next(self.packets)
            
            # if packet is null, send eos frame and jump to status IDLE
            shape = (self.stream.width, self.stream.height)
            if packet.buffer_size <= 0:
                # send eos frame
                frame = Frame(packet, shape, is_last=True, context=self.context)

                # after send eos, it jump to status END
                self._status = Status.END
            else:
                self._frame = self._frame + 1
                frame = Frame(packet, shape, frame_id=self._frame, context=self.context)

            # do video decode
            self.decoder.process(frame)
            return True

        elif self._status == Status.END:
            if self.decoder.image.empty():
                self.decoder.finish()
                self._status = Status.IDLE
                return False

            return True
        else:
            self._status = Status.IDLE
            return False


    def read(self, print_status=True):     
        """Read one frame from caputure in device. If we cann't get the data in the timeout, 
            it will be raise an timeout error.
        Args:
            print_status (bool, optional): Get the queue status for True

        Returns:
            [AscendArray]: Get a decoded frame.
        """
        if self._status != Status.IDLE and not self.decoder.image.empty():
            if print_status:
                Log(INFO, f"qsize = {self.decoder.image.qsize()}.")
                print(f"qsize = {self.decoder.image.qsize()}")
            frame_id, image = self.decoder.image.get(timeout=30)
            
            return image, frame_id
        else:
            Log(WARNING, 'read image failed in Video Capture.')
            return (None, None)

    def set(self, attr, value):
        """set(attr, value) -> retval
            Sets a property in the VideoCapture
        Args:
            attr (str): Property from VideoCapture Properties (eg. 'width', 'fps', ...)
            value (int): Value of the property

        Returns:
            bool : `True` if the property is supported by the backend used by the VideoCapture instance.
        """
        if self._status == Status.READY:
            if attr == 'qsize': 
                self.decoder.queue_size = value
            elif attr == 'channel_id': 
                self.decoder.channel_id = value
            elif attr == 'ref_num': 
                self.decoder.ref_num = value
            elif attr == 'enc_type': 
                self.decoder.encoder_type = value
            elif attr == 'pix_fmt': 
                self.decoder.pic_format = value
            elif attr == 'bit_depth': 
                self.decoder.bit_depth = value
            elif attr == 'out_mode': 
                self.decoder.out_mode = value
            else:
                Log(ERROR, f'attr {attr} is not support in VideoCapture.')
                return False
            return True
        else:
            Log(ERROR, f'Set attr {attr} in status {self._status}.')
            raise ValueError(f"Set attr {attr} in status {self._status}.")

    def get(self, attr):   
        """ get(attr) -> retval
            Returns the specified VideoCapture property
        Args:
            attr (str): Property from VideoCapture Properties (eg. 'width', 'fps', ...)
            
        Returns:
            [Value]: Value for the specified property. Value 0 is returned when querying a 
                property that is not supported by the backend used by the VideoCapture instance.
        """
        if not isinstance(attr, str):
            raise TypeError(f"Input attr expects a string, but got {type(attr)}.")
        
        attr_dict = {
            'width'       : self.stream.width,
            'height'      : self.stream.height,
            'frame_height': self.stream.coded_height,
            'frame_width' : self.stream.coded_width,
            'aspect_ratio': self.stream.display_aspect_ratio,
            'fps'         : self.framerate,
            'format'      : self.stream.format,
            'pix_fmt'     : self.stream.pix_fmt, 
            'frames'      : self.stream.frames
        }

        try:
            return attr_dict[attr]
        except KeyError:
            return 0

    def release(self):
        """Closes video file or capturing device and release resource.
        """
        if hasattr(self, 'container'):
            self.container.close()

        if hasattr(self, 'decoder'):
            del self.decoder
        
        self._status = False

    def __del__(self):
        self.release()

if __name__ == "__main__":
    import cv2
    from resource.context import Context
    resource = Context({1})
    context = resource.context_dict[1]
    stream_path = './cars_around_mountain_640_360.264'

    cap = VideoCapture(context, stream_path)

    while cap.is_open():
        image, frame_id = cap.read()
        if image:
            yuv_np = image.to_np
            img_color = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_NV21)
            cv2.imshow('result', img_color)
            cv2.waitKey(10)
    cv2.destroyAllWindows()