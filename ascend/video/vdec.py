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
import sys
import numpy as np
from queue import Queue

from ..resource.mem import malloc
from ..resource.context import bind_context
from ..data.ascendarray import AscendArray
from ..common.log import Log
from ..common.const import *
from ..common.align import aligned, calc_size
from .frame import Frame


class Vdec():
    def __init__(self, 
                context, 
                channel=0, 
                en_type=0, 
                pix_fmt=PIXEL_FORMAT_YUV_SEMIPLANAR_420, 
                timeout=50):
        self.class_name = self.__class__.__name__
        self.__check_para(context, channel, en_type)
        self._vdec_exit = True

        # video channel parameter
        self.context     = context
        self._channel_id = channel
        self._en_type    = en_type
        self._format     = pix_fmt

        # video decoder queue, to save decoded image. Default queue size is 25.
        self.image = Queue(maxsize=25)
        self.qsize = 25

        self._bind_thread = self.__init_thread(timeout=timeout)
        self.channel_desc = self.__vdec_create_channel()
        self._frame_conf = self.__frame_config()

    def __check_para(self, context, channel, en_type):
        """ check input parameters of instance Vdec.
        Args:
            context : the device context resource.
            string  : input stream.
            channel : the channel id of Vdec.
            en_type : the encode type of stream.

        Returns:
            None.
        """
        # bind context
        bind_context(context)

        if channel < 0 or channel > 31:
            raise ValueError(f"Configurated channel_id is out of range [0 31].")

        if en_type not in [H265_MAIN_LEVEL, H264_BASELINE_LEVEL, H264_MAIN_LEVEL, H264_HIGH_LEVEL]:
            raise ValueError(f"Configurated en_type is invalid.")

    def __init_thread(self, timeout=100):
        """ initial a thread to watch the callback.
        Args:
            timeout: 

        Returns:
            None.
        """
        cb_thread_id, ret = acl.util.start_thread(self.__thread_func, [timeout])
        if ret != ACL_SUCCESS:
            raise ValueError(f"Vdec initial acl thread failed, return {ret}.")
        
        return cb_thread_id

    def __thread_func(self, args_list):
        # bind context resource
        bind_context(self.context)

        timeout = args_list[0]
        while self._vdec_exit:
            acl.rt.process_report(timeout)

        Log(INFO, 'vdec_thread_func out')

    def __callback(self, input_stream_desc, output_pic_desc, user_data):
        """inner callback function of video decoder.
        Args:
            input_stream_desc, output_pic_desc, user_data

        Returns:
            None.
        """
        # step 1: release input stream desc and free input memory
        if input_stream_desc:
            ret = acl.media.dvpp_destroy_stream_desc(input_stream_desc)
            if ret != ACL_SUCCESS:
                Log(WARNING, f'destroy input_stream_desc failed, return {ret}.')

        # step 2: save the decode image and release desc
        if output_pic_desc:
            if self.image.qsize() == self.qsize - 1:
                Log(WARNING, f'buffer queue {self.image.queue()} is almost full.')

            ret = acl.media.dvpp_get_pic_desc_ret_code(output_pic_desc)
            if ret != ACL_SUCCESS:
                Log(WARNING, f'image decode error, return {ret}.')
            else:
                size = acl.media.dvpp_get_pic_desc_size(output_pic_desc)
                buffer = acl.media.dvpp_get_pic_desc_data(output_pic_desc)
                align_w = acl.media.dvpp_get_pic_desc_width_stride(output_pic_desc)

                frame_id = user_data[0]
                yuv_image = AscendArray((align_w,), np.dtype('uint8'), size=size, buffer=buffer,\
                                         format=self._format, flag='DEVICE')

                self.image.put((frame_id, yuv_image.reshape(yuv_image.shape[::-1])), timeout=30)
                self._channel_id = user_data[1]

            ret = acl.media.dvpp_destroy_pic_desc(output_pic_desc)
            if ret != 0:
                Log(WARNING, f'destroy pic desc failed, return {ret}.')
        
        Log(INFO, f'frame:{user_data[0]} vdec decoding frame success.')          
                

    def __vdec_create_channel(self):
        """ create a vdec channel, and bind it with a thread and other info.
        Args:
            None

        Returns:
            channel_desc : the created channel desc resource
        """
        channel_desc = acl.media.vdec_create_channel_desc()

        ret = acl.media.vdec_set_channel_desc_channel_id(channel_desc, self._channel_id)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc channel id, return {ret}."
        
        ret = acl.media.vdec_set_channel_desc_thread_id(channel_desc, self._bind_thread)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc thread id, return {ret}."

        ret = acl.media.vdec_set_channel_desc_callback(channel_desc, self.__callback)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc call back, return {ret}."

        ret = acl.media.vdec_set_channel_desc_entype(channel_desc, self._en_type)
        assert ret == ACL_SUCCESS, f"Failed to set channel en_type, return {ret}."

        ret = acl.media.vdec_set_channel_desc_out_pic_format(channel_desc, self._format)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc format, return {ret}."

        ret = acl.media.vdec_create_channel(channel_desc)
        assert ret == ACL_SUCCESS, f"Failed to create channel desc, return {ret}."

        Log(INFO, 'Vdec init success')
        return channel_desc

    def __frame_config(self):
        """create a frame config.
        Args:
            None

        Returns:
            frame_conf.
        """
        frame_conf = acl.media.vdec_create_frame_config()

        Log(INFO, 'frame config success in vdec init.')
        return frame_conf 

    def __stream_desc(self, frame):
        """ create a stream desc and bind to input frame data pointer and size info.
        Args:
            frame : input frame packet
        Returns:
            stream_desc: the created stream desc.
        """
        stream_desc = acl.media.dvpp_create_stream_desc()

        # if the last frame, send an Eos frame to vdec
        if frame.is_last:
            Log(INFO, 'Start to send EOS frame.')
            ret = acl.media.dvpp_set_stream_desc_eos(stream_desc, 1)
            if ret != ACL_SUCCESS:
                raise ValueError(f"Failed to set EOS to input stream desc, return {ret}.")
            return stream_desc
    
        ret = acl.media.dvpp_set_stream_desc_data(stream_desc, frame.data)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set stream desc data, return {ret}.")

        ret = acl.media.dvpp_set_stream_desc_size(stream_desc, frame.size)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set stream desc size, return {ret}.")
        return  stream_desc

    def __pic_desc(self, width, height):
        """ create a output image desc and bind memory and other info.
        Args:
            height  : output image's width
            width   : output image's height
        Returns:
            pic_desc: the created image desc.
        """
        aligned_w = aligned(width, aligned='w')
        aligned_h = aligned(height, aligned='h')

        buffer_size = calc_size(aligned_w, aligned_h, self._format)
        buffer = malloc(buffer_size, flag='DVPP')
                
        # create picture desc resource
        pic_desc = acl.media.dvpp_create_pic_desc()

        ret = acl.media.dvpp_set_pic_desc_data(pic_desc, buffer)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set pic desc data, return {ret}.")

        ret = acl.media.dvpp_set_pic_desc_size(pic_desc, buffer_size)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set pic desc size, return {ret}.")

        ret = acl.media.dvpp_set_pic_desc_format(pic_desc, self._format)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set pic desc format, return {ret}.")

        return pic_desc

    def process(self, frame):
        """Read one packet frame from caputure in host and send to dvpp.
        Args:
            frame : a frame packet with data and frame info

        Returns:
            None.
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"Input frame expects Frame instance, but got {type(frame)}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel_desc must be created before using in process.")

        if not hasattr(self, '_frame_conf'):
            raise ValueError(f"vdec _frame_conf must be created before using in process.")
        # configurate input stream
        stream_desc = self.__stream_desc(frame)

        # configure output picture
        pic_desc = self.__pic_desc(frame.width, frame.height)

        user_data = (frame.frame_id, self._channel_id)
        # send one frame to vdec
        ret = acl.media.vdec_send_frame(self.channel_desc,
                                        stream_desc,
                                        pic_desc,
                                        self._frame_conf,
                                        user_data)
        if ret != ACL_SUCCESS:
            Log(WARNING, 'vdec send frame failed.')
        
    def finish(self):
        """ finish bind thread working. 

        .. warning:: 
            This function should be called after sending last eos-frame.
        """
        self._vdec_exit = False
            
        ret = acl.util.stop_thread(self._bind_thread)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to stop bind thread {self._bind_thread}, return {ret}.")


    @property
    def queue_size(self):
        return self.image.qsize()

    @queue_size.setter
    def queue_size(self, maxsize):
        if not self.image.empty():
            raise ValueError(f"Original queue should be empty before configuring.")
        self.qsize = maxsize
        self.image = Queue(maxsize=maxsize)

    @property
    def channel_id(self):
        return self._channel_id

    @channel_id.setter
    def channel_id(self, channel_id):
        if not isinstance(channel_id, int):
            raise TypeError(f"Configurate channel_id expects an int, but got {type(channel_id)}.")
        
        if channel_id < 0 or channel_id > 31:
            raise ValueError(f"Configurate channel_id is out of range [0 31].")
        
        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel desc must be created before set channel_id.")

        ret = acl.media.vdec_set_channel_desc_channel_id(self.channel_desc, channel_id)
        assert ret == ACL_SUCCESS, f"Failed to set channel id, return {ret}."
        self._channel_id = channel_id

    @property
    def encoder_type(self):
        return self._en_type

    @encoder_type.setter
    def encoder_type(self, encoder_type):
        if not isinstance(encoder_type, int):
            raise TypeError(f"Configurate encoder_type expects an int, but got {type(encoder_type)}.")

        if encoder_type < 0 or encoder_type > 4:
            raise ValueError(f"Configurate encoder_type is out of range [0 3] \
                    (0:H265_MAIN_LEVEL, 1:H264_BASELINE_LEVEL, 2:H264_MAIN_LEVEL, 3:H264_HIGH_LEVEL).")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel desc must be created before set encoder_type.")

        ret = acl.media.vdec_set_channel_desc_entype(self.channel_desc, encoder_type)
        assert ret == ACL_SUCCESS, f"Failed to set channel en_type, return {ret}."
        self._en_type = encoder_type

    @property
    def pic_format(self):
        return self._format

    @pic_format.setter
    def pic_format(self, pic_format):
        if not isinstance(pic_format, int):
            raise TypeError(f"Configurate encoder_type expects an int, but got {type(pic_format)}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel desc must be created before set pic_format.")

        ret = acl.media.vdec_set_channel_desc_out_pic_format(self.channel_desc, pic_format)
        assert ret == ACL_SUCCESS, f"Failed to set channel pic_format, return {ret}."
        self._format = pic_format

    @property
    def ref_num(self):
        ref_frame_num = acl.media.vdec_get_channel_desc_ref_frame_num(self.channel_desc)
        return ref_frame_num

    @property
    def out_mode(self):
        out_mode, ret = acl.media.vdec_get_channel_desc_out_mode(self.channel_desc)
        assert ret == ACL_SUCCESS, f"Failed to get channel out_mode, return {ret}."
        return out_mode

    @out_mode.setter
    def out_mode(self, out_mode):
        if not isinstance(out_mode, int):
            raise TypeError(f"Configure out_mode expects an int, but got {type(out_mode)}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel desc must be created before set out_mode.")

        ret = acl.media.vdec_set_channel_desc_out_mode(self.channel_desc, out_mode)
        if ret != ACL_SUCCESS:
            raise ValueError(f"vdec set channel out_mode failed, return {ret}.")
        

    @property
    def bit_depth(self):
        bit_depth = acl.media.vdec_get_channel_desc_bit_depth(self.channel_desc)
        return bit_depth
    
    @bit_depth.setter
    def bit_depth(self, bit_depth):
        if not isinstance(bit_depth, int):
            raise TypeError(f"input bit_depth expects an int, but got {type(bit_depth)}.")

        if bit_depth != 0 and bit_depth != 1:
            raise TypeError(f"input bit_depth expects 0 or 1, but got {bit_depth}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel desc must be created before set out_mode.")

        ret = acl.media.vdec_set_channel_desc_bit_depth(self.channel_desc, bit_depth)
        if ret != ACL_SUCCESS:
            raise ValueError(f"vdec set bit_depth failed, return {ret}.")

    def __del__(self):
        # bind context to release
        bind_context(self.context)
        
        if hasattr(self, 'channel_desc'):
            ret = acl.media.vdec_destroy_channel(self.channel_desc)
            if ret != ACL_SUCCESS:
                raise ValueError(f"Failed to destroy vdec channel, return {ret}.")
            
            ret = acl.media.vdec_destroy_channel_desc(self.channel_desc)
            if ret != ACL_SUCCESS:
                raise ValueError(f"Failed to destroy vdec channel desc, return {ret}.")
        
        if hasattr(self, '_frame_conf'):
            ret = acl.media.vdec_destroy_frame_config(self._frame_conf)
            if ret != ACL_SUCCESS:
                raise ValueError(f"Failed to destroy frame config, return {ret}.")
        
        self._vdec_exit = False
        
        if hasattr(self, '_bind_thread'):
            ret = acl.util.stop_thread(self._bind_thread)
            if ret != ACL_SUCCESS:
                raise ValueError(f"Failed to stop bind thread {self._bind_thread}, return {ret}.")
        
        while not self.image.empty():
            data = self.image.get()
            del data


if __name__ == "__main__":
    stream_addr = '../../tests/data/vdec_h265_1frame_rabbit_1280x720.h265'
    from resource.context import Context

    resource = Context({1})
    context = resource.context_dict[1]
    decoder = Vdec(context, channel=0)

    img = np.fromfile(stream_addr, dtype=np.uint8)
    img_buffer_size = img.size
    img_ptr = acl.util.numpy_to_ptr(img)

    img_device, ret = acl.media.dvpp_malloc(img_buffer_size)

    ret = acl.rt.memcpy(img_device,
                        img_buffer_size,
                        img_ptr,
                        img_buffer_size,
                        ACL_MEMCPY_HOST_TO_DEVICE)
    if ret != ACL_SUCCESS:
        print("memcpy error.")

    for i in range(5):
        frame = Frame(img_device, 1280, 720, img_buffer_size)
        decoder.process(frame)
        if len(decoder.image) > 0:
            frame_id, image = decoder.image.popleft()
            yuv = image.to_np

    acl.media.dvpp_free(img_device)
    del decoder
    del resource

        

