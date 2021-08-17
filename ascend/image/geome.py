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
import numbers
import os.path as osp
from pathlib import Path

from ..common.const import *
from ..data.ascendarray import AscendArray
from ..resource.mem import Memory
from ..resource.context import create_stream
from ..common.log import Log
from ..common.align import aligned, calc_size
from ..common.clip import (_scale_size, bbox_clip, bbox_scaling)
from ..common.path import check_file_exist, is_str, mkdir_or_exist


try:
    from PIL import Image
except ImportError:
    Image = None


def align_img(img, func='vdec'):
    """Align a Ascend image's width and height using this function.

    Args:
        img (AscendArray): Image or croped area of image(bbox).
        func (str, optional): The alinged method to be used. Defaults to 'vdec'.

    Returns:
        [tuple]: The returned data format is `(image, (w, h))`
    """
    if not isinstance(img, AscendArray):
        raise TypeError(f"Input img expects an AscendArray, but got {type(img)}.")

    # step 1: get the truth h and w of image.
    if img.format in yuv420:
        h, w = (img.shape[0] * 2 // 3, img.shape[1])
    else:
        h, w = img.shape[:2]

    # step 2: caculate image aligned shape and buffer size
    stride_w = aligned(w, func=func, aligned='w')
    stride_h = aligned(h, func=func, aligned='h')
    buf_size = calc_size(stride_w, stride_h, img.format)

    # step 3: if image not aligned, create a new image(a AscendArray object)
    if (stride_h, stride_w) != (h, w):
        image = AscendArray((stride_w,), np.dtype('uint8'), size=buf_size, \
                        format=img.format, flag='DVPP')
        img.to(image)
        image.reshape(image.shape[::-1])
        return image, (w, h)

    return img, (w, h)

def create_align_img(w, h, format=PIXEL_FORMAT_YUV_SEMIPLANAR_420, func='vdec'):
    """Assemble image parameters to the created image desc.

    Args:
        w (int): The original image's width.
        h (int): The original image's height.
        format (int, optional): Image pixel formats. Defaults to `PIXEL_FORMAT_YUV_SEMIPLANAR_420`.
        func (str, optional): Ascend image process method. Defaults to 'vdec'.

    Returns:
        AscendArray: The created image
    """
    # step 1: caculate image aligned w,h and buffer size
    stride_w = aligned(w, func=func, aligned='w')
    stride_h = aligned(h, func=func, aligned='h')
    buf_size = calc_size(stride_w, stride_h, format)

    # step 2: create a image(a AscendArray object) with dvpp memory
    image = AscendArray((stride_w,), np.dtype('uint8'), size=buf_size, \
                        format=format, flag='DVPP')
    image.reshape(image.shape[::-1])

    return image



def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag == 'color':
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag == 'grayscale':
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale" or "unchanged", '
                f'but got {flag}')
    return array

class PicDesc():
    """Define a PicDesc object, to construct an picture info.

    Attributes:
        desc (obj): The created dvpp pic desc

    """
    def __init__(self, obj, shape=None):
        if not isinstance(obj, (AscendArray, list)):
            raise TypeError(f"Input obj expects an AscendArray object or \
                a list of AscendArray object, but got {type(obj)}.")

        if shape and not isinstance(shape, tuple):
            raise TypeError(f"Input shape expects an tuple, but got {type(shape)}.")

        if isinstance(obj, AscendArray):
            self._desc = acl.media.dvpp_create_pic_desc()
            self.__assemble_desc(self._desc, obj, shape=shape)
        else:
            batch = len(obj)
            assert batch > 0, f"Input image list is null."

            self._batch_desc = acl.media.dvpp_create_batch_pic_desc(batch)
            for i, image in enumerate(obj):
                desc = acl.media.dvpp_get_pic_desc(self._batch_desc, i)
                self.__assemble_desc(desc, image, shape=shape)

    def __assemble_desc(self, desc, image, shape=None):
        """assemble image parameters to the created image desc.

        Args:
            desc  : the created pic desc
            image : (AsendArray) image or croped area of image(bbox).
            shape : the original size of image or (w, h) not to be aligned.

        Returns:
            None
        """
        if not isinstance(image, AscendArray):
            raise TypeError(f"Input image expects an AscendArray object, but got {type(image)}.")

        h, w = (image.shape[0] * 2 // 3, image.shape[1]) if image.format in yuv420 else image.shape[:2]

        if shape is None:
            shape = (w, h)

        ret = acl.media.dvpp_set_pic_desc_data(desc, image.ascend_data)
        assert ret == ACL_SUCCESS, f"Failed to set dvpp pic desc data, return {ret}."

        ret = acl.media.dvpp_set_pic_desc_size(desc, image.nbytes)
        assert ret == ACL_SUCCESS, f"Failed to set dvpp pic desc size, return {ret}."

        ret = acl.media.dvpp_set_pic_desc_format(desc, image.format)
        assert ret == ACL_SUCCESS, f"Failed to set dvpp pic desc format, return {ret}."

        ret = acl.media.dvpp_set_pic_desc_width(desc, shape[0])
        assert ret == ACL_SUCCESS, f"Failed to set dvpp pic desc width, return {ret}."

        ret = acl.media.dvpp_set_pic_desc_height(desc, shape[1])
        assert ret == ACL_SUCCESS, f"Failed to set dvpp pic desc height, return {ret}."

        ret = acl.media.dvpp_set_pic_desc_width_stride(desc, w)
        assert ret == ACL_SUCCESS, f"Failed to set dvpp pic desc width_stride, return {ret}."

        ret = acl.media.dvpp_set_pic_desc_height_stride(desc, h)
        assert ret == ACL_SUCCESS, f"Failed to set dvpp pic desc height_stride, return {ret}."

    @property
    def desc(self):
        if hasattr(self, '_desc'):
            return self._desc
        
        if hasattr(self, '_batch_desc'):
            return self._batch_desc
        

    def __del__(self):
        if hasattr(self, '_desc'):
            ret = acl.media.dvpp_destroy_pic_desc(self._desc)
            assert ret == ACL_SUCCESS, f"destroy pic desc error in \
                dvpp_destroy_pic_desc, return {ret}."

        if hasattr(self, '_batch_desc'):
            ret = acl.media.dvpp_destroy_batch_pic_desc(self._batch_desc)
            assert ret == ACL_SUCCESS, f"destroy pic desc error in \
                dvpp_destroy_batch_pic_desc, return {ret}."


class Image():
    """Define an Image class to process image.

    Attributes:
       context (int): The context resource working on
       stream (int): The stream resource working on


    .. hint:: 
       - imdecode  : reads an image from the specified numpy ndarray and decode to yuv image
       - imresize  : resizes the image img down to or up to the specified size
       - imrescale : Resize image while keeping the aspect ratio.
       - imflip    : Flip an image horizontally or vertically
       - imrotate  : Rotate an image
       - imcrop    : Crop image patches

    """
    def __init__(self, context=None, stream=None):
        self.class_name = self.__class__.__name__
        self.context = context
 
        # create a stream according to context
        self.stream = create_stream(context)

        # create dvpp image processing channel
        self._channel_desc = acl.media.dvpp_create_channel_desc()
        ret = acl.media.dvpp_create_channel(self._channel_desc)
        if ret != ACL_SUCCESS:
            raise ValueError(f"create channel failed, return {ret}.")

    def imdecode(self, array, format=PIXEL_FORMAT_YUV_SEMIPLANAR_420, return_shape=False):
        """ imdecode(array, format) -> retval
        
        .. note::
            The function imdecode reads an image from the specified numpy ndarray and decode to yuv image.
            See www.hiascend.com for the list of supported format.

        Args:
            array (ndarray): Input numpy ndarray.
            format (int, optional): The supported decode image format. Defaults to PIXEL_FORMAT_YUV_SEMIPLANAR_420.
            return_shape (bool, optional): Return original shape(h, w, c) of image.

        Returns:
            AscendArray : The decoded image(AscendArray obj)

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        data = np.fromfile('./image.jpg', dtype=np.uint8)
        image = Img.imdecode(data)
        ```
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Input array expects an np.ndarray object, but got {type(array)}.")

        img_ptr = acl.util.numpy_to_ptr(array)

        w, h, c, ret = acl.media.dvpp_jpeg_get_image_info(img_ptr, array.nbytes)
        if ret == ACL_SUCCESS:
            decoder = 'jpg'
        else:
            w, h, c, ret = acl.media.dvpp_png_get_image_info(img_ptr, array.nbytes)
            assert ret ==ACL_SUCCESS, f"Get input image's info failed, return {ret}."
            decoder = 'png'

        constrain_ = 8192 if decoder == 'jpg' else 4096
        if w < 32 or w > constrain_:
            raise ValueError(f"Input image's width out of range [32, {constrain_}].")

        if h < 32 or h > constrain_:
            raise ValueError(f"Input image's height out of range [32, {constrain_}].")

        if decoder == 'jpg' and format not in [
            PIXEL_FORMAT_YUV_SEMIPLANAR_420,
            PIXEL_FORMAT_YVU_SEMIPLANAR_420,
            PIXEL_FORMAT_YVU_SEMIPLANAR_422,
            PIXEL_FORMAT_YVU_SEMIPLANAR_444
            ]:
            raise ValueError(f"Input decode {format} is invalid, this format is unsupport.")

        if decoder =='png' and format not in [
            PIXEL_FORMAT_RGB_888,
            PIXEL_FORMAT_BGR_888,
            PIXEL_FORMAT_ARGB_8888,
            PIXEL_FORMAT_ABGR_8888,
            PIXEL_FORMAT_RGBA_8888,
            PIXEL_FORMAT_BGRA_8888
            ]:
            raise ValueError(f"Input decode {format} is invalid, this format is unsupport.")

        if self._channel_desc is None:
            raise ValueError(f"channel desc is not initialized before imdecode.")

        # clone input image to device
        data = AscendArray.clone(array, context=self.context, flag="DVPP")

        image = create_align_img(w, h, format=format, func='jpegd' if decoder == 'jpg' else 'pngd')

        # create output image desc and bind it with a output AscendArray object
        pic_inst = PicDesc(image, (w, h))

        # do jpeg decode
        if decoder == 'jpg':
            ret = acl.media.dvpp_jpeg_decode_async(self._channel_desc,
                                                data.ascend_data,
                                                data.nbytes,
                                                pic_inst.desc,
                                                self.stream)
            assert ret == ACL_SUCCESS, f"Failed to do dvpp_jpeg_decode_async, return {ret}."
        else:
            ret = acl.media.dvpp_png_decode_async(self._channel_desc,
                                                data.ascend_data,
                                                data.nbytes,
                                                pic_inst.desc,
                                                self.stream)
            assert ret == ACL_SUCCESS, f"Failed to do dvpp_png_decode_async, return {ret}."
        
        # finish jpeg image decode
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to decode jpeg image in synchronize_stream, return {ret}.")

        # release the unused resource in future
        del (data, pic_inst)
        Log(INFO, 'imdecode success.')
        if return_shape:
            return image, (h, w, c)
        else:
            return image


    def imencode(self, ext, img, params=None):
        """ Encodes an image into a memory buffer.
         
        .. note::
            The function imencode compresses the image and stores it in the memory buffer 
            that is resized to fit the result. See www.hiascend.com for the list of 
            supported format and flags description.
         
        Args:
            ext (str): File extension that defines the output format.
            img (AscendArray): Image to be written
            params (str, optional): Format-specific parameters, not use.

        Returns:
            AscendArray : Output buffer resized to fit the compressed image.
        
        Typical usage example:
        ```python
        img_encode = img.imencode('.jpg', yuv_img)
        str_encode = np.array(img_encode).tostring()
        ```
        """
        if ext not in IMG_EXT_ENC:
            raise TypeError(f"Input ext expects {IMG_EXT_ENC}, but got {ext}.")

        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray object, but got {type(img)}.")

        if img.format not in [
            PIXEL_FORMAT_YUV_SEMIPLANAR_420,
            PIXEL_FORMAT_YVU_SEMIPLANAR_420,
            PIXEL_FORMAT_YUYV_PACKED_422, 
            PIXEL_FORMAT_UYVY_PACKED_422, 
            PIXEL_FORMAT_YVYU_PACKED_422,
            PIXEL_FORMAT_VYUY_PACKED_422
            ]:
            raise ValueError(f"Encode img's format {img.format} is invalid, this format not support.")

        if self._channel_desc is None:
            raise ValueError(f"channel desc is not initialized before imdecode.")

        src_img, shape = align_img(img, func='encode')

        # create input image desc and bind it with a input AscendArray object
        src_inst = PicDesc(src_img, shape)

        # create a jpeg image encode configurate
        config = acl.media.dvpp_create_jpege_config()
        ret = acl.media.dvpp_set_jpege_config_level(config, 100)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Set jpege config level failed, return {ret}.")

        # predict out buffer size according to image desc
        buff_size, ret = acl.media.dvpp_jpeg_predict_enc_size(src_inst.desc, config)
        if ret != ACL_SUCCESS:
            raise ValueError(f"jpeg predict encode size failed, return {ret}.")

        # config out buffer of encode
        out_size = np.array([buff_size], dtype=np.int32)
        size_ptr = acl.util.numpy_to_ptr(out_size)
        enc_jpg = AscendArray((buff_size,), np.dtype('uint8'), format=img.format, flag='DVPP')
        
        # do jpeg decode
        ret = acl.media.dvpp_jpeg_encode_async(self._channel_desc,
                                               src_inst.desc,
                                               enc_jpg.ascend_data,
                                               size_ptr,
                                               config,
                                               self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_jpeg_encode_async, return {ret}.")
        
        # finish jpeg image decode
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to encode jpeg image in synchronize_stream, return {ret}.")

        # trans AscendArray data to numpy array
        jpeg_img = enc_jpg.to_np

        # release the unused resource in future
        del (src_inst, src_img, enc_jpg)
        Log(INFO, 'imdecode success.')

        ret = acl.media.dvpp_destroy_jpege_config(config)
        if ret != ACL_SUCCESS:
            raise ValueError(f"destroy jpege config failed, return {ret}.")
        return jpeg_img


    def imresize(self, img, dsize, interpolation='huawei'):
        """imresize(array, img, dsize, interpolation='huawei') -> retval

        To shrink an image, it will generally look best with #INTER_AREA interpolation

        Args:
            img (AscendArray): Input image
            dsize (tuple): The output image size (tuple(w, h));
            interpolation (str, optional): The interpolation method, its default method is researched 
                by ourself.. Defaults to 'huawei'.

        Returns:
            AscendArray: The resized image

        .. note::
            The function resize resizes the image img down to or up to the specified size. Note that the
            initial dst type or size are not taken into account. Instead, the size and type are derived from
            the img, dsize. 

        you may call the function as follows:
        ```python
        # explicitly specify dsize (tuple[int]): Target size (w, h).
        imresize(img, dsize, interpolation)
        ```
        If you want to decimate the image by factor of 2 in each direction, you can call the function this way:
        ```python
        # specify the element w and h of the destination image size.
        imresize(img, (w//2, h//2), interpolation)
        ```
        """
        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray object, but got {type(img)}.")

        # valid format is in range [PIXEL_FORMAT_YUV_400 -> PIXEL_FORMAT_BGRA_8888]
        if img.format < 0 or img.format > 17:
            raise ValueError(f"Input image format {img.format} is invalid, only support \
                [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)].")

        # get the truth h and w of image.
        if img.format in yuv420:
            h, w = (img.shape[0] * 2 // 3, img.shape[1])
        else:
            h, w = img.shape[:2]

        if dsize and not isinstance(dsize, tuple):
            raise TypeError(f"Input dsize expects a tuple object, but got {type(dsize)}.")

        if dsize and (dsize[0] > 16 * w or dsize[0] < int(1/32 * w)):
            raise ValueError(f"Input dsize[0] out of resize ratio [1/32, 16]")

        if dsize and (dsize[1] > 16 * h or dsize[1] < int(1/32 * h)):
            raise ValueError(f"Input dsize[1] out of resize ratio [1/32, 16]")

        if interpolation not in interp_codes.keys():
            raise ValueError(f"Input decode flag {interpolation} is unsupport.")

        if self._channel_desc is None:
            raise ValueError(f"channel desc is not initialized before imdecode.")

        # do image aligned and create aligned image to save resized image
        src_img, shape = align_img(img, func='resize')
        dst_img = create_align_img(dsize[0], dsize[1], format=img.format, func='resize')

        # create output image desc and bind it with a output AscendArray object
        src_inst = PicDesc(src_img, shape)
        dst_inst = PicDesc(dst_img, dsize)

        # create resize config
        resize_conf = acl.media.dvpp_create_resize_config()

        # set resize interpolation
        ret = acl.media.dvpp_set_resize_config_interpolation(resize_conf, interp_codes[interpolation])
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to set resize_config_interpolation, return {ret}.")

        # do image resize
        ret = acl.media.dvpp_vpc_resize_async(self._channel_desc,
                                              src_inst.desc,
                                              dst_inst.desc,
                                              resize_conf,
                                              self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_vpc_resize_async, return {ret}.")

        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to resize image in synchronize_stream, return {ret}.")

        ret = acl.media.dvpp_destroy_resize_config(resize_conf)
        if ret != ACL_SUCCESS:
            raise ValueError(f"do dvpp_destroy_resize_config failed, return {ret}.")

        del (img, src_img, src_inst, dst_inst)
        Log(INFO, 'resize image success')

        return dst_img


    def imrescale(self, img, scale, interpolation='huawei', return_shape=False):
        """Resize image while keeping the aspect ratio.

        Args:
            img (AsecendArray): The input image.
            scale (float, tuple[int]): The scaling factor or maximum size. If it is a float 
                number, then the image will be rescaled by this factor, else if it is a tuple 
                of 2 integers, then the image will be rescaled as large as possible within 
                the scale.
            interpolation (str, optional): Same as :func:`imresize`.
            return_shape (bool, optional): Return scale shape not aligned.

        Returns:
            AsecendArray : The rescaled image.

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        yuv_rescale1 = Img.imrescale(yuv, 0.3)
        yuv_rescale2 = Img.imrescale(yuv, (320, 540))
        ```
        """
        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray object, but got {type(img)}.")

        # valid format is in range [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)]
        if img.format < 0 or img.format > 17:
            raise ValueError(f"Input image format {img.format} is invalid, only support \
                [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)].")

        # get the truth h and w of image.
        if img.format in yuv420:
            h, w = (img.shape[0] * 2 // 3, img.shape[1])
        else:
            h, w = img.shape[:2]

        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')  
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)

            # find the maximum rescale ratio, and the resized shape should be valid.
            scale_factor = min(max_long_edge/max(h, w), max_short_edge/min(h, w))
        else:
            raise TypeError(
                f'Scale must be a number or tuple of int, but got {type(scale)}')

        # caculate new rescaled size according to scale_factor 
        new_size = _scale_size((w, h), scale_factor)

        # do resize
        rescaled_img = self.imresize(img, new_size, interpolation=interpolation)

        if return_shape:
            return rescaled_img, new_size
        else:
            return rescaled_img

    def imflip(self, img, direction='horizontal'):
        """Flip an image horizontally or vertically.

        Args:
            img (AscendArray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or "vertical" or "diagonal".

        Returns:
            AscendArray: The flipped image.
        
        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        im_flip = Img.imflip(img, direction='horizontal')
        ```
        """
        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray object, but got {type(img)}.")

        # valid format is in range [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)]
        if img.format < 0 or img.format > 17:
            raise ValueError(f"Input image format {img.format} is invalid, only support \
                [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)].")

        assert direction in ['horizontal', 'vertical', 'diagonal']

        arr = img.to_np

        if direction == 'horizontal':
            return AscendArray.clone(np.flip(arr, axis=1))
        elif direction == 'vertical':
            return AscendArray.clone(np.flip(arr, axis=0))
        else:
            return AscendArray.clone(np.flip(arr, axis=(0, 1)))


    def imrotate(self, 
                img,
                angle,
                center=None,
                scale=1.0,
                border_value=0,
                interpolation='bilinear',
                auto_bound=False):
        """Rotate an image.

        Args:
            img (AscendArray): Image to be rotated.
            angle (float): Rotation angle in degrees, positive values mean clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the rotation in the source 
                image. If not specified, the center of the image will be used.
            scale (float, optional): Isotropic scale factor.
            border_value (int, optional): Border value.
            interpolation (str, optional): Same as function: `resize`.
            auto_bound (bool, optional): Whether to adjust the image size to cover the whole
                rotated image.

        Returns:
            AscendArray: The rotated image.
        """
        try:
            import cv2
            cv2_interp_codes = {
                'nearest': cv2.INTER_NEAREST,
                'bilinear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC,
                'area': cv2.INTER_AREA,
                'lanczos': cv2.INTER_LANCZOS4
            }
        except ImportError:
            Log(ERROR, 'import cv2 error while using imrotate, you should install opencv first.')
            return

        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img.to_np,
            matrix, (w, h),
            flags=cv2_interp_codes[interpolation],
            borderValue=border_value)
        return AscendArray.clone(rotated)
        

    def imcrop(self, img, bboxes, scale=1.0):
        """Crop image patches, and resize bboxes.

        Args:
            img (AscendArray): Image to be cropped.
            bboxes (ndarray): Shape (k, 4) or (4,), location of cropped bboxes.
            scale (float, optional): Scale ratio of bboxes, the default value 1.0 means no padding.

        Returns:
            list[AscendArray] or AscendArray: The cropped image patches.

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        bboxes = np.array([[20, 40, 159, 259],[400, 200, 479, 419]], dtype=int)
        yuv_croped = Img.imcrop(img, bboxes)
        ```
        """
        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray, but we got a {type(img)}.")

        # valid format is in range [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)]
        if img.format < 0 or img.format > 17:
            raise ValueError(f"Input image format {img.format} is invalid, only support \
                [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)].")

        if not isinstance(bboxes, np.ndarray):
            raise TypeError(f"Input bbox expects an np.ndarray, but we got a {type(bboxes)}.")

        if self._channel_desc is None:
            raise ValueError(f"channel desc is not initialized before imdecode.")

        # step 1: caculate the cliped bbox shape
        _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
        clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

        # align original image
        src_img, shape = align_img(img, func='crop')

        # step 2: input desc assembling, only process one image.
        image_inst = PicDesc([src_img], shape)
        roi_n_list = [clipped_bbox.shape[0]]

        # step 3: clip bboxes of image
        bboxes, roi_list = [], []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = clipped_bbox[i, :].tolist()
            
            # align w and h, and calc memory size of output box image
            dst_stride_w = aligned(x2 - x1, aligned='w')
            dst_stride_h = aligned(y2 - y1, aligned='h')
            dst_buf_size = calc_size(dst_stride_w, dst_stride_h, img.format)
            
            # create a bbox(a AscendArray object) and bind with dvpp memory
            dst_image = AscendArray((dst_stride_w,), np.dtype('uint8'), size=dst_buf_size, \
                                format=img.format, flag='DVPP')
            bboxes.append(dst_image.reshape(dst_image.shape[::-1]))

            # create roi description
            roi_conf = acl.media.dvpp_create_roi_config(x1, x2, y1, y2)
            roi_list.append(roi_conf)

        # create output image desc and bind it with a output AscendArray object
        bboxes_inst = PicDesc(bboxes)

        # step 4: crop processing
        _, ret = acl.media.dvpp_vpc_batch_crop_async(
                                            self._channel_desc,
                                            image_inst.desc, 
                                            roi_n_list,
                                            bboxes_inst.desc,
                                            roi_list,
                                            self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_vpc_crop_async, return {ret}.")

        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to crop image in synchronize_stream, return {ret}.")
        
        for roi in roi_list:
            ret = acl.media.dvpp_destroy_roi_config(roi)
            assert ret == ACL_SUCCESS, \
                ValueError(f"Failed to do dvpp_destroy_resize_config, return {ret}.")

        del image_inst
        del bboxes_inst
        Log(INFO, 'vpc crop process success')

        if not np.isclose(scale, 1.0, rtol=1e-2, atol=1e-03):
            res_img = []
            for bbox in bboxes:
                res_img.append(self.imrescale(bbox, scale))
            return res_img

        return bboxes

    def bbox_resize(self, img, bboxes, sizes, inner_resize=True):
        """Clip the bboxes image and resize to specified size.

        2 steps:  clip the bboxes image -> resize croped image.

        Args:
            img (AscendArray): Image to be cliped.
            bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
            sizes (int): Shape (k, 2) or (2, ), scale sizes of bboxes.
            inner_resize (bool, optional): Use dvpp auto resize function to resize croped boxes

        Returns:
            list[AscendArray] or AscendArray: The cropped and resized image patches.

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        bboxes = np.array([[20, 40, 159, 259],[400, 200, 479, 419]], dtype=int)
        sizes = np.array([[300, 300], [400, 400]])
        yuv_croped = Img.bbox_resize(img, bboxes, sizes)
        ```
        """
        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray, but we got {type(img)}.")

        # valid format is in range [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)]
        if img.format < 0 or img.format > 17:
            raise ValueError(f"Input image format {img.format} is invalid, only support \
                [PIXEL_FORMAT_YUV_400(0) -> PIXEL_FORMAT_BGRA_8888(17)].")

        if not isinstance(bboxes, np.ndarray):
            raise TypeError(f"Input bboxes expects an np.ndarray, but given {type(bboxes)}.")

        if not isinstance(sizes, np.ndarray):
            raise TypeError(f"Input sizes expects an np.ndarray, but given {type(sizes)}.")

        if self._channel_desc is None:
            raise ValueError(f"channel desc is not initialized before imdecode.")

        assert sizes.shape[-1] == 2, f"Input sizes dim 2 must be 2."
        if sizes.ndim == 2 and sizes.shape[0] != bboxes.shape[0]:
            raise ValueError(f"Input sizes's shape expects same to bboxes.")

        # step 1: caculate the cliped bbox shape
        _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = bbox_scaling(_bboxes, 1.0).astype(np.int32)
        clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

        # align original image 
        src_img, shape = align_img(img, func='crop')

        # step 2: input desc assembile, only process one image.
        image_inst = PicDesc([src_img], img.shape[::-1])
        roi_n_list = [clipped_bbox.shape[0]]

        # step 3: clip bboxes of image
        bboxes, roi_list = [], []
        _sizes = sizes[None, ...].tolist() if sizes.ndim == 1 else sizes.tolist()
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = clipped_bbox[i, :].tolist()

            # align w and h, and calc memory size of output box image
            if inner_resize:
                dst_stride_w = aligned(_sizes[i][0], aligned='w')
                dst_stride_h = aligned(_sizes[i][1], aligned='h')
            else:
                dst_stride_w = aligned(x2 - x1, aligned='w')
                dst_stride_h = aligned(y2 - y1, aligned='h')
            dst_buf_size = calc_size(dst_stride_w, dst_stride_h, img.format)
            
            # create a bbox(a AscendArray object) and bind with dvpp memory
            dst_image = AscendArray((dst_stride_w,), np.dtype('uint8'), size=dst_buf_size, \
                                format=img.format, flag='DVPP')
            bboxes.append(dst_image.reshape(dst_image.shape[::-1]))

            # create roi description
            roi_conf = acl.media.dvpp_create_roi_config(x1, x2, y1, y2)
            roi_list.append(roi_conf)

        # create output image desc and bind it with a output AscendArray object
        bboxes_inst = PicDesc(bboxes)

        # step 4: crop processing
        _, ret = acl.media.dvpp_vpc_batch_crop_async(
                                            self._channel_desc,
                                            image_inst.desc, 
                                            roi_n_list,
                                            bboxes_inst.desc,
                                            roi_list,
                                            self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_vpc_crop_async, return {ret}.")

        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to crop image in synchronize_stream, return {ret}.")
        
        for roi in roi_list:
            ret = acl.media.dvpp_destroy_roi_config(roi)
            assert ret == ACL_SUCCESS, \
                ValueError(f"Failed to do dvpp_destroy_resize_config, return {ret}.")

        del (image_inst, bboxes_inst)
        Log(INFO, 'vpc crop process success')

        if not inner_resize:
            res_img = []
            for bbox in bboxes:
                res_img.append(self.imresize(bbox, _sizes[i]))
            return res_img

        return bboxes

    def __align_roi(self, bbox, img_shape):
        """ align bbox left-top and right-bottom point
        Args:
            bbox : Input bbox(np.ndrray) with shape (4, ) or (n, 4)
            img_shape : tuple(h, w), input image shape, and bbox location should in image.

        Returns:
            aligned bbox, list with shape (4, ) or (n, 4).
        """
        if not isinstance(img_shape, tuple):
            raise TypeError(f"Input img_shape expect a tuple, but got {type(img_shape)}.")
            
        bbox = bbox[None, ...] if bbox.ndim == 1 else bbox
        for i, box in enumerate(bbox):
            assert box[0] < img_shape[1] and box[1] < img_shape[0] and box[2] > 0 \
                and box[3] > 0, f"Input {i}-th bbox([startx, starty, endx, endy]) beyond the image."

        for i in range(bbox.shape[0]):
            bbox[i][0] = bbox[i][0] >> 1 << 1
            bbox[i][1] = bbox[i][1] >> 1 << 1
            bbox[i][2] = (bbox[i][2] >> 1 << 1) - 1
            bbox[i][3] = (bbox[i][3] >> 1 << 1) - 1

        return bbox

    def imcrop_paste(self, src_img, dst_img, crop_bbox, paste_bbox):
        """Crop a bbox from src image and paste the cropped bbox to dst image

        Args:
            src_img (AscendArray): Input src image(AscendArray) to be croped.
            dst_img (AscendArray): Input dst image(AscendArray) to be pasted.
            crop_bbox (np.ndarray): Shape (4, ) or (n, 4), location of cropped bboxe, data arrangement,
                np.array([startx, starty, endx, endy], dtype=int).
            paste_bbox (np.ndarray): Shape (4, ) or (n, 4), location of pasted bboxe, data arrangement
                same to crop_bbox.

        Returns:
            AscendArray: Dest image after cropping and pasting.

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        crop_bbox = np.array([40, 30, 140, 230], dtype='int32')
        paste_bbox = np.array([70, 80, 170, 280], dtype='int32')
        Img.imcrop_paste(yuv_src, yuv_dst, crop_bbox, paste_bbox)
        ```
        """

        if not isinstance(src_img, AscendArray):
            raise TypeError(f"Input src_img expects an AscendArray, but we got a {type(src_img)}.")

        if not isinstance(dst_img, AscendArray):
            raise TypeError(f"Input dst_img expects an AscendArray, but we got a {type(dst_img)}.")

        if not isinstance(crop_bbox, np.ndarray):
            raise TypeError(f"Input bbox expects an np.ndarray, but we got a {type(crop_bbox)}.")

        if not isinstance(paste_bbox, np.ndarray):
            raise TypeError(f"Input bbox expects an np.ndarray, but we got a {type(crop_bbox)}.")

        if crop_bbox.shape[-1] != 4 or paste_bbox.shape[-1] != 4:
            raise ValueError(f"Input crop_bbox or paste_bbox column dim should be 4.")

        if crop_bbox.shape[0] != paste_bbox.shape[0]:
            raise ValueError(f"Input crop_bbox and paste_bbox should have same number bbox.")

        # step 1: check input bbox, make startx and starty to even and endx and endy to odd
        cbbox = self.__align_roi(crop_bbox, src_img.shape).tolist()
        pbbox = self.__align_roi(paste_bbox, dst_img.shape).tolist()

        # step 2: align original image
        crop_img, _ = align_img(src_img, func='crop')
        paste_img, _ = align_img(dst_img, func='crop')

        # step 3: create output image desc and bind it with a output AscendArray object
        src_desc = PicDesc([crop_img], src_img.shape)
        dst_desc = PicDesc([paste_img] * len(pbbox), dst_img.shape)

        # step 4: startx of paste box aligned to 16 and append crop/paste roi to list
        crop_conf_list, past_conf_list = [], []
        for i in range(len(pbbox)):
            pbbox[i][0] = aligned(pbbox[i][0], aligned='w')
            cstartx, cendx, cstarty, cendy = cbbox[i][0], cbbox[i][2], cbbox[i][1], cbbox[i][3]
            pstartx, pendx, pstarty, pendy = pbbox[i][0], pbbox[i][2], pbbox[i][1], pbbox[i][3]
            crop_conf = acl.media.dvpp_create_roi_config(cstartx, cendx, cstarty, cendy)
            past_conf = acl.media.dvpp_create_roi_config(pstartx, pendx, pstarty, pendy)
            crop_conf_list.append(crop_conf)
            past_conf_list.append(past_conf)

        # step 5: do crop and paste
        _, ret = acl.media.dvpp_vpc_batch_crop_and_paste_async(self._channel_desc,
                                                            src_desc.desc,
                                                            [len(pbbox)], 
                                                            dst_desc.desc,
                                                            crop_conf_list, 
                                                            past_conf_list,
                                                            self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_vpc_batch_crop_and_paste_async, return {ret}.")

        # step 6: synchronize stream and finish crop and paste 
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to crop and paste image in synchronize_stream, return {ret}.")

        for conf in crop_conf_list:
            ret = acl.media.dvpp_destroy_roi_config(conf)
            assert ret == ACL_SUCCESS, f"Failed to do dvpp_destroy_resize_config, return {ret}."

        for conf in past_conf_list:
            ret = acl.media.dvpp_destroy_roi_config(conf)
            assert ret == ACL_SUCCESS, f"Failed to do dvpp_destroy_resize_config, return {ret}."

        del (crop_img, src_desc, dst_desc)
        Log(INFO, 'crop and paste process success')
        return paste_img

    def impad(self, img, shape=None, padding=None, pad_val=0, padding_mode='constant'):
        """Pad the given image to a certain shape or pad on all sides with
            specified padding mode and padding value.

        Args:
            img (AscendArray): Image to be padded.
            shape (tuple[int]): Expected padding shape (h, w). Default: None.
            padding (int or tuple[int]): Padding on each border. If a single int is provided this 
                is used to pad all borders. If tuple of length 2 is provided this is the padding 
                on left/right and top/bottom respectively. If a tuple of length 4 is provided 
                this is the padding for the [left, top, right and bottom] borders respectively.
                Default: None. Note that `shape` and `padding` can not be both set.
            pad_val (Number): Values to be filled in padding areas when padding_mode is 
                'constant'. Default: 0.
            padding_mode (str): Type of padding. Only support: constant. Default: constant.

                - constant: pads with a constant value, this value is specified with pad_val.

        Returns:
            AscendArray : The padded image.

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        yuv_pad = Img.impad(img, padding=(20, 50, 100, 200), pad_val=128)
        ```
        """
        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray, but we got a {type(img)}.")

        if shape and not isinstance(shape, tuple):
            raise TypeError(f"Input shape expects an tuple, but we got a {type(shape)}.")

        if padding and not isinstance(padding, (int, tuple)):
            raise TypeError(f"Input padding expects an int or tuple, but we got a {type(padding)}.")

        if padding_mode not in ['constant']:
            raise TypeError(f"Input padding_mode expects in ['constant'], but we got {padding_mode}.")

        assert (shape is not None) ^ (padding is not None), \
                f"Input `shape` and `padding` can not be both set."

        # get the truth h and w of image.
        if img.format in yuv420:
            h, w = (img.shape[0] * 2 // 3, img.shape[1])
        else:
            h, w = img.shape[:2]

        if shape is not None:
            padding = (0, 0, shape[1] - w, shape[0] - h)

        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                            f'But received {padding}')

        # align original image shape and create a output to save padding result.
        src_img, _ = align_img(img)
        dst_w = padding[0] + w + padding[2]
        dst_h = padding[1] + h + padding[3]
        dst_img = create_align_img(dst_w, dst_h, format=img.format)

        # set the padding result's memory with pad_val
        Memory.reset(dst_img.ascend_data, dst_img.nbytes, pad_val)

        # create output image desc and bind it with a output AscendArray object
        src_desc = PicDesc(src_img, img.shape)
        dst_desc = PicDesc(dst_img, (dst_h, dst_w))

        # paste area startx, starty, endx, endy, and align startx with 16
        startx = aligned(padding[0], aligned='w')
        starty = padding[1] >> 1 << 1
        endx = ((padding[0] + w) >> 1 << 1) - 1
        endy = ((padding[1] + h) >> 1 << 1) - 1

        # create roi description
        roi_conf = acl.media.dvpp_create_roi_config(0, (w >> 1 << 1) - 1, 0, (h >> 1 << 1) - 1)
        pst_conf = acl.media.dvpp_create_roi_config(startx, endx, starty, endy)

        ret = acl.media.dvpp_vpc_crop_and_paste_async(self._channel_desc,
                                                      src_desc.desc,
                                                      dst_desc.desc,
                                                      roi_conf,
                                                      pst_conf,
                                                      self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_vpc_crop_and_paste_async, return {ret}.")

        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to crop and paste image in synchronize_stream, return {ret}.")

        ret = acl.media.dvpp_destroy_roi_config(roi_conf)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_destroy_resize_config, return {ret}.")

        ret = acl.media.dvpp_destroy_roi_config(pst_conf)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to do dvpp_destroy_resize_config, return {ret}.")

        del (src_img, src_desc, dst_desc)
        Log(INFO, 'image padding process success')
        return dst_img

    def imread(self, img_or_path, flag='color', channel_order='bgr', backend='ascend'):
        """Read an image.

        Args:
            img_or_path (ndarray or str or Path): Either a numpy array or str or pathlib.Path. If 
                it is a numpy array (loaded image), then it will be returned as is.
            flag (str): Flags specifying the color type of a loaded image, candidates are `color`, 
                `grayscale` and `unchanged`. It only work for `cv2` and `pillow`.
            channel_order (str): Order of channel, candidates are `bgr` and `rgb`, not work for ascend.
            backend (str): The image decoding backend type. Options are `cv2`, `pillow`, `ascend`. 
                Default: `ascend`.

        Returns:
            AscendArray: Loaded image array.

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        yuv = Img.imread('./image.jpg')
        ```
        """
        if backend not in ['cv2', 'pillow', 'ascend']:
            raise ValueError(f'backend: {backend} is not supported. Supported "backends are '
                            "'cv2', 'turbojpeg', 'pillow'.")

        if backend == 'cv2':
            try:
                import cv2
                imread_flags = {
                    'color': cv2.IMREAD_COLOR,
                    'grayscale': cv2.IMREAD_GRAYSCALE,
                    'unchanged': cv2.IMREAD_UNCHANGED
                }
            except ImportError:
                Log(ERROR, "import cv2 error while using imread with backend 'cv2'.")
                return

        if isinstance(img_or_path, Path):
            img_or_path = str(img_or_path)

        if isinstance(img_or_path, np.ndarray):            
            decode_img = self.imdecode(img_or_path)
            return decode_img
        elif is_str(img_or_path):
            check_file_exist(img_or_path, f'img file does not exist: {img_or_path}')
            if backend == 'pillow':
                img = Image.open(img_or_path)
                img = _pillow2array(img, flag, channel_order)
                return AscendArray.clone(img)
            else:
                flag = imread_flags[flag] if is_str(flag) else flag
                img = cv2.imread(img_or_path, flag)
                if flag == IMREAD_COLOR and channel_order == 'rgb':
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                return AscendArray.clone(img)
        else:
            raise TypeError('"img" must be a numpy array or a str or a pathlib.Path object')


    def imwrite(self, img, file_path, params=None, auto_mkdir=True):
        """Write image to file.

        Args:
            img (AscendArray): Image array to be written.
            file_path (str): Image file path.
            params (None, str or list): If params=='pillow', save image with PIL. if params is list, 
                this same as opencv's func: `imwrite` interface, only work for opencv.
            auto_mkdir (bool): If the parent folder of `file_path` does not exist,
                whether to create it automatically.

        Returns:
            bool: Successful or not.

        Typical usage example:
        ```python
        Img = ascend.Image(ctx)
        Img.imwrite(yuv_org, './saved_img.jpg')  
        ```
        """
        if not isinstance(img, AscendArray):
            raise TypeError(f"Input img expects an AscendArray, but got {type(img)}.")
        
        if auto_mkdir:
            dir_name = osp.abspath(osp.dirname(file_path))
            mkdir_or_exist(dir_name)
        
        if params is None:
            ext = '.png' if file_path.endswith('.png') or file_path.endswith('.PNG')  else '.jpg'
            file = self.imencode(ext, img)
            with open(file, 'w+') as fw:
                fw.write(file)
        elif params == 'pillow':
            pil_image = Image.fromarray(img.to_np)
            pil_image.save(file_path)
        else:
            try:
                import cv2
                cv2.imwrite(file_path, img.to_np, params)
            except ImportError:
                Log(ERROR, f"import cv2 error while using imwrite save image with params {params}.")


    def __del__(self):
        if hasattr(self, '_channel_desc'):
            ret = acl.media.dvpp_destroy_channel(self._channel_desc)
            assert ret == ACL_SUCCESS, f"dvpp destroy channel failed, return {ret}."

            ret = acl.media.dvpp_destroy_channel_desc(self._channel_desc)
            assert ret == ACL_SUCCESS, f"dvpp destroy channel desc failed, return {ret}."


if __name__ == "__main__":
    import pdb
    from resource.context import Context
    context = Context({0})
    ctx = context.context_dict[0]
    img = Image(ctx)
    pdb.set_trace()

    data = np.fromfile('girl1.jpg', dtype=np.uint8)
    yuv_img = img.imdecode(data)

    img_encode = img.imencode('.jpg', yuv_img)
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()

    with open('img_encode.jpg', 'wb') as f:
        f.write(str_encode)
        f.flush

    del img, context


