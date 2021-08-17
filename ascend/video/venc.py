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
import time
import numpy as np

from ..common.log import Log
from ..common.const import *
from ..common.path import is_filepath
from ..resource.mem import memcpy_d2h
from ..resource.context import bind_context
from .frame import Frame

class EncFrame():
    """Define a EncFrame class to manage encode frame

    Attributes:
        frame_conf (obj): initial frame config
        eos (obj): create an eos frame
    """
    def __init__(self):
        self.class_name = self.__class__.__name__
        self.frame_conf = acl.media.vdec_create_frame_config()

        # set frame to not eos
        acl.media.venc_set_frame_config_eos(self.frame_conf, 0)
        acl.media.venc_set_frame_config_force_i_frame(self.frame_conf, 0)

        Log(INFO, 'frame config success')

    def eos(self):
        """create Eos frame config.
        Args:
            None

        Returns:
            None.
        """
        Log(INFO, 'Start to send EOS frame.')
        ret = acl.media.venc_set_frame_config_eos(self.frame_conf, 1)
        assert ret == ACL_SUCCESS, f"Failed to set EOS to input pic desc, return {ret}."

        ret = acl.media.venc_set_frame_config_force_i_frame(self.frame_conf, 0)
        assert ret == ACL_SUCCESS, f"Failed to set force i frame, return {ret}."

        return ret
        
    def __del__(self):
        if hasattr(self, 'frame_conf'):
            ret = acl.media.vdec_destroy_frame_config(self.frame_conf)
            assert ret == ACL_SUCCESS, f"Failed to destroy frame config, return {ret}."
        

class Venc():
    """Define a Venc class to encode the frame using dvpp.

    Attributes:
        context (int): context resource of this module working on
        stream_dir (str): a string or Path to save the encoded stream
        width (int): the encoding frame's width
        height (int): the encoding frame's height
        en_type (int): encode type of dvpp
        frame_itval (int): key frame interval, and the defaut value is 25
        pix_fmt (int): the frame's pixel format, only support yuv420 NV12/NV21
        timeout (int): the encoding thread time out of dvpp 
        rc_mode (int): set/get frame number of video stream
        fps (int): set/get frame per second of encoding stream 
        bit_rate (int): set/get bit rate
 
    """
    def __init__(self, 
                context, 
                stream_dir,
                width,
                height,
                en_type=0, 
                frame_interval = 25,
                pix_fmt=PIXEL_FORMAT_YUV_SEMIPLANAR_420, 
                timeout=50):
        self.class_name = self.__class__.__name__
        self._venc_exit = True

        # video parameter
        self.context     = context
        self.stream_dir  = stream_dir
        self._width      = width
        self._height     = height
        self._en_type    = en_type
        self._format     = pix_fmt
        self._frame_itval= frame_interval
        self.__check_para()

        self._bind_thread = self.__init_thread(timeout=timeout)
        self.channel_desc = self.__venc_create_channel()
        self._en_frame = EncFrame()

    def __check_para(self):
        """ check input parameters of instance Vdec.
        Args:
            None

        Returns:
            None.
        """
        # bind context
        bind_context(self.context)

        if not is_filepath(self.stream_dir):
            raise ValueError(f"Input stream_dir:{self.stream_dir} is not exist.")

        if self._width < 128 or self._width > 1920:
            raise ValueError(f"Input width only support [0, 1920].")

        if self._height < 128 or self._height > 1920:
            raise ValueError(f"Input height only support [0, 1080].")

        if self._en_type not in [H265_MAIN_LEVEL, H264_BASELINE_LEVEL, H264_MAIN_LEVEL, H264_HIGH_LEVEL]:
            raise ValueError(f"Configurated en_type is invalid.")

        if self._format not in [PIXEL_FORMAT_YUV_SEMIPLANAR_420, PIXEL_FORMAT_YVU_SEMIPLANAR_420]:
            raise ValueError(f"Input pix_fmt only support yuv420 and yvu420.")

        if self._frame_itval < 1 or self._frame_itval > 65536:
            raise ValueError(f"Input frame_interval only support [1, 65536].")

    def __init_thread(self, timeout=100):
        """ initial a thread to watch the callback.
        Args:
            timeout: 

        Returns:
            None.
        """
        cb_thread_id, ret = acl.util.start_thread(self.__thread_func, [self.context, timeout])
        if ret != ACL_SUCCESS:
            raise ValueError(f"Venc initial acl thread failed, return {ret}.")
        
        return cb_thread_id

    def __thread_func(self, args_list):
        context = args_list[0]
        timeout = args_list[1]

        # bind context
        bind_context(context)
        
        while self._venc_exit:
            acl.rt.process_report(timeout)

        Log(INFO, 'vdec_thread_func out')


    def __callback(self, input_pic_desc, output_stream_desc, user_data):
        """inner callback function of video encoder.
        Args:
            input_pic_desc, output_tream_desc, user_data

        Returns:
            None.
        """
        # step 1: release input stream desc and free input memory
        if output_stream_desc == 0:
            Log(INFO, 'output_stream_desc is null in callback.')
            return

        stream_data = acl.media.dvpp_get_stream_desc_data(output_stream_desc)
        if stream_data is None:
            Log(INFO, 'stream_data is None in func dvpp_get_stream_desc_data.')
            return

        ret = acl.media.dvpp_get_stream_desc_ret_code(output_stream_desc)
        if ret != ACL_SUCCESS:
            Log(INFO, f'dvpp_get_stream_desc_ret_code in callback is {ret}.')
            return 

        stream_size = acl.media.dvpp_get_stream_desc_size(output_stream_desc)
        
        # step 2: numpy array to save encode data
        enc_data = np.zeros(stream_size, dtype=np.byte)
        enc_dptr = acl.util.numpy_to_ptr(enc_data)

        # copy device data to host
        memcpy_d2h(enc_dptr, stream_data, stream_size)

        # step 3: write encode stream
        with open(self.stream_dir, 'ab') as f:
            f.write(enc_data)
        
    def __venc_create_channel(self):
        """ create a venc channel, and band it with a thread and other info.
        Args:
            None

        Returns:
            channel_desc : the created channel desc resource
        """
        channel_desc = acl.media.venc_create_channel_desc()

        ret = acl.media.venc_set_channel_desc_thread_id(channel_desc, self._bind_thread)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc thread id, return {ret}."

        ret = acl.media.venc_set_channel_desc_callback(channel_desc, self.__callback)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc call back, return {ret}."

        ret = acl.media.venc_set_channel_desc_entype(channel_desc, self._en_type)
        assert ret == ACL_SUCCESS, f"Failed to set channel en_type, return {ret}."

        ret = acl.media.venc_set_channel_desc_pic_format(channel_desc, self._format)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc format, return {ret}."

        ret = acl.media.venc_set_channel_desc_key_frame_interval(channel_desc, self._frame_itval)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc frame interval, return {ret}."

        ret = acl.media.venc_set_channel_desc_pic_height(channel_desc, self._height)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc width, return {ret}."

        ret = acl.media.venc_set_channel_desc_pic_width(channel_desc, self._width)
        assert ret == ACL_SUCCESS, f"Failed to set channel desc height, return {ret}."

        ret = acl.media.venc_create_channel(channel_desc)
        assert ret == ACL_SUCCESS, f"Failed to create channel desc, return {ret}."

        Log(INFO, '__venc_create_channel init success')
        return channel_desc



    def __pic_desc(self, frame):
        """ create a input image desc and bind memory and other info.
        Args:
            frame : input frame
        Returns:
            pic_desc: the created picture desc.
        """
        pic_desc = acl.media.dvpp_create_pic_desc()

        # if the last frame, send an Eos frame to venc
        if frame.is_last:
            return self._en_frame.eos()

        ret = acl.media.dvpp_set_pic_desc_data(pic_desc, frame.data)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set stream desc data, return {ret}.")

        ret = acl.media.dvpp_set_pic_desc_size(pic_desc, frame.size)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set stream desc size, return {ret}.")
        return  pic_desc

    def finish(self):
        """ create a eos stream desc and set it to vdec.
        
        .. warning::
                this function should be called after processing all valid frame.
        Args:
            None
        Returns:
            pic_desc: the created picture desc.
        """
        # step 1: set finish configure
        stream_desc = acl.media.dvpp_create_stream_desc()

        # stream_desc set function
        ret = acl.media.dvpp_set_stream_desc_format(stream_desc, self._en_type)
        assert ret == ACL_SUCCESS, f"set stream desc format failed, return {ret}." 

        ret = acl.media.dvpp_set_stream_desc_timestamp(stream_desc, int(time.time()))
        assert ret == ACL_SUCCESS, f"set stream desc timestamp failed, return {ret}."

        ret = acl.media.dvpp_set_stream_desc_ret_code(stream_desc, 1)
        assert ret == ACL_SUCCESS, f"set stream desc ret code failed, return {ret}."

        ret = acl.media.dvpp_set_stream_desc_eos(stream_desc, 1)
        assert ret == ACL_SUCCESS, f"set stream desc eos failed, return {ret}."

        ret = acl.media.dvpp_destroy_stream_desc(stream_desc)
        assert ret == ACL_SUCCESS, f"destroy stream desc failed, return {ret}."

        # step 2: stop thread
        self._venc_exit = False
        
        ret = acl.util.stop_thread(self._bind_thread)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to stop band thread {self._bind_thread}, return {ret}.")


    def process(self, frame):
        """Assemble one frame and send to video encoder
        Args:
            frame : a frame with data and frame info

        Returns:
            None.
        """
        if not isinstance(frame, Frame):
            raise TypeError(f"Input frame expects Frame instance, but got {type(frame)}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel_desc must be created before using in process.")

        # configurate input picture description
        pic_desc = self.__pic_desc(frame)

        # send one frame to venc
        ret = acl.media.venc_send_frame(self.channel_desc,
                                        pic_desc,
                                        0,
                                        self._en_frame.frame_conf,
                                        None)
        if ret != ACL_SUCCESS:
            Log(WARNING, 'vdec send frame failed.')

        Log(INFO, "venc send frame success.")

    @property
    def width(self):
        width = acl.media.venc_get_channel_desc_pic_width(self.channel_desc)
        return width

    @width.setter
    def width(self, width):
        if width < 128 or width > 1920:
            raise ValueError(f"Input width expects in range [128, 1920], but got {width}.")

        ret = acl.media.venc_set_channel_desc_pic_width(self.channel_desc, width)
        assert ret == ACL_SUCCESS, f"set channel desc pic width failed, return {ret}."
        self._width = width

    @property
    def height(self):
        height = acl.media.venc_get_channel_desc_pic_height(self.channel_desc)
        return height

    @height.setter
    def height(self, height):
        if height < 128 or height > 1920:
            raise ValueError(f"Input height expects in range [128, 1920], but got {height}.")

        ret = acl.media.venc_set_channel_desc_pic_height(self.channel_desc, height)
        assert ret == ACL_SUCCESS, f"set channel desc pic height failed, return {ret}."
        self._height = height

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

        ret = acl.media.venc_set_channel_desc_entype(self.channel_desc, encoder_type)
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

        ret = acl.media.venc_set_channel_desc_pic_format(self.channel_desc, pic_format)
        assert ret == ACL_SUCCESS, f"Failed to set channel pic_format, return {ret}."
        self._format = pic_format

    @property
    def frame_itval(self):
        return self._frame_itval

    @frame_itval.setter
    def frame_itval(self, frame_itval):
        if not isinstance(frame_itval, int):
            raise TypeError(f"Configure frame_itval expects an int, but got {type(frame_itval)}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"vdec channel desc must be created before set frame_itval.")

        ret = acl.media.venc_set_channel_desc_key_frame_interval(self.channel_desc, frame_itval)
        if ret != ACL_SUCCESS:
            raise ValueError(f"vdec set channel frame_itval failed, return {ret}.")
        

    @property
    def fps(self):
        fps = acl.media.venc_get_channel_desc_src_rate(self.channel_desc)
        return fps
    
    @fps.setter
    def fps(self, fps):
        if not isinstance(fps, int):
            raise TypeError(f"input fps expects an int, but got {type(fps)}.")

        if fps < 1 or fps > 120:
            raise TypeError(f"input fps expects in [1, 120], but got {fps}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"venc channel desc must be created before set fps.")

        ret = acl.media.venc_set_channel_desc_src_rate(self.channel_desc, fps)
        if ret != ACL_SUCCESS:
            raise ValueError(f"venc set fps failed, return {ret}.")

    @property
    def rc_mode(self):
        rc_mode = acl.media.venc_get_channel_desc_rc_mode(self.channel_desc)
        return rc_mode

    @rc_mode.setter
    def rc_mode(self, rc_mode):
        if rc_mode < 0 or rc_mode > 2:
            raise TypeError(f"input rc_mode expects in [0, 2], but got {rc_mode}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"venc channel desc must be created before set rc_mode.")

        ret = acl.media.venc_set_channel_desc_rc_mode(self.channel_desc, rc_mode)
        if ret != ACL_SUCCESS:
            raise ValueError(f"venc set rc_mode failed, return {ret}.")

    @property
    def bit_rate(self):
        bit_rate = acl.media.venc_get_channel_desc_max_bit_rate(self.channel_desc)
        return bit_rate

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        if bit_rate < 10 or bit_rate > 30000:
            raise TypeError(f"input bit_rate expects in [10, 30000], but got {bit_rate}.")

        if not hasattr(self, 'channel_desc'):
            raise ValueError(f"venc channel desc must be created before set bit_rate.")

        ret = acl.media.venc_set_channel_desc_max_bit_rate(self.channel_desc, bit_rate)
        if ret != ACL_SUCCESS:
            raise ValueError(f"venc set bit_rate failed, return {ret}.")


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


if __name__ == "__main__":
    import pdb
    stream_dir = './test_venc_h264.264'
    from resource.context import Context
  
    context = Context({1}).context_dict[1]
    Img = ascend.Image(context)
        
    src_img = np.fromfile('./image/xiaoxin.jpg', dtype=np.uint8)
    yuv_src = Img.imdecode(src_img)
    yuv_resize = Img.imresize(src_img, (1280, 720))
    encode = Venc(context, stream_dir, 1280, 720)

    pdb.set_trace()
    for i in range(5):
        if i == 4:
            frame = Frame(yuv_resize, is_last=True, context=context)
        else:
            frame = Frame(yuv_resize)
        encode.process(frame)

    encode.finish()
    del encode

        

