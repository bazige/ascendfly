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
from ..common.clip import bbox_clip
from ..model.model import AscendModel
from ..resource.context import create_stream


ACL_DATA_WITH_DYNAMIC_AIPP = 2


class Aipp():
    """ define a Aipp object, to manage image color space convert and pixel process.

    Attributes:
        context (int): context resource
        shape (tuple[int]): the original image shape(tuple(w, h))
        format (str or int): the output image format

    Methods:
        desc    : the created dvpp pic desc
    """

    def __init__(self, context, model_inst, tensor_name, **kwargs):
        if not isinstance(context, int):
            raise TypeError(
                f"Input context expects an int value, but got {type(context)}.")

        if not isinstance(model_inst, AscendModel):
            raise TypeError(
                f"Input model_inst expects an AscendModel, but got {type(context)}.")

        if not isinstance(tensor_name, str):
            raise TypeError(
                f"Input tensor_name expects a string, but got {type(tensor_name)}.")

        self._stream = create_stream(context)

        # get input id that setting aipp
        need_aipp = []
        for index in range(model_inst.net_in_n):
            aipp_type, _, ret = acl.mdl.get_aipp_type(
                model_inst.model_id, index)
            assert ret == ACL_SUCCESS, f"get aipp type failed, return {ret}."

            if aipp_type == ACL_DATA_WITH_DYNAMIC_AIPP:
                need_aipp.append(index)

        assert len(need_aipp) >= 2, f"aipp only support one input, but got {len(need_aipp)}."

        self._shape = None
        self._format = None
        self._swap_switch = False
        self._swap_ax_switch = False
        batch_size = model_inst.tensor[tensor_name].shape[0]
        self.aipp_set = acl.mdl.create_aipp(batch_size)

        for key, arg_val in kwargs.items():
            if key == 'shape':
                self.shape = arg_val
            elif key == 'format':
                self.format = arg_val
            elif key == 'csc':
                # key-value is a 3-elements tuple(csc_mat, i_bias_mat, o_bias_mat)
                csc_mat = arg_val[0]
                i_bias_mat = arg_val[1]
                o_bias_mat = arg_val[2]
                self.set_csc(self, csc_mat, i_bias_mat, o_bias_mat)
            elif key == 'swap_switch':
                self.swap_switch = arg_val
            elif key == 'swap_ax':
                self.swap_ax = arg_val
            elif key == 'means':
                self.means(arg_val, input_id=need_aipp[0])
            elif key == 'min':
                self.set_means(arg_val, input_id=need_aipp[0])
            elif key == 'var':
                self.set_var(arg_val, input_id=need_aipp[0])
            elif key == 'crop':
                self.set_crop(arg_val, input_id=need_aipp[0])
            elif key == 'set':
                self.set_aipp(model_inst.model_id,
                              model_inst.dataset, input_id=need_aipp[0])

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        """ Set aipp input image shape.

        Args:
            shape : The input image shape.

        Returns:
            None.
        """
        if not isinstance(shape, tuple):
            raise TypeError(
                f"Input shape expects a tuple, but got {type(shape)}.")

        assert len(shape) == 2, f"Input shape expects 2 elements tuple, but got {len(shape)}."
        ret = acl.mdl.set_aipp_src_image_size(self.aipp_set, shape[0], shape[1])
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set aipp input image format failed, return {ret}.")

        self._shape = shape

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, fmt):
        """ Set aipp input image format.

        Args:
            fmt : The input image format.

        Returns:
            None.
        """
        if not isinstance(fmt, int):
            raise TypeError(f"Input ")

        try:
            format = pix_aipp_map[fmt]
        except:
            raise ValueError(f"Aipp does not support this format {fmt}.")

        ret = acl.mdl.set_aipp_input_format(self.aipp_set, format)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set aipp input image format failed, return {ret}.")

        self._format = format

    def set_csc(self, csc_mat, i_bias_mat, o_bias_mat):
        """ Set aipp color space convert parameters.

        Args:
            csc_mat : The input color space convert matrix.

        Returns:
            None.
        """
        if not isinstance(csc_mat, np.ndarray):
            raise TypeError(
                f"Input csc_mat expects a np.ndrray, but got {type(csc_mat)}.")

        if not isinstance(i_bias_mat, np.ndarray):
            raise TypeError(
                f"Input bias_mat expects a np.ndrray, but got {type(i_bias_mat)}.")

        if not isinstance(o_bias_mat, np.ndarray):
            raise TypeError(
                f"Input bias_mat expects a np.ndrray, but got {type(o_bias_mat)}.")

        assert csc_mat.shape == (3, 3), \
            f"Input csc_mat expects a 3x3 matrix, but got {csc_mat.shape} matrix."

        assert i_bias_mat.shape[-1] == 3, \
            f"Input bias_mat expects a 3x1 matrix, but got a {i_bias_mat.shape} matrix."

        assert o_bias_mat.shape[-1] == 3, \
            f"Input bias_mat expects a 3x1 matrix, but got a {o_bias_mat.shape} matrix."

        # convert to 1 dim array
        i_bias_mat = i_bias_mat.flatten()
        o_bias_mat = o_bias_mat.flatten()

        # matrix_r0c0 matrix_r0c1 matrix_r0c2
        # matrix_r1c0 matrix_r1c1 matrix_r1c2
        # matrix_r2c0 matrix_r2c1 matrix_r2c2
        ret = acl.mdl.set_aipp_csc_params(self.aipp_set, 1,
                                          csc_mat[0][0], csc_mat[0][1], csc_mat[0][2],
                                          csc_mat[1][0], csc_mat[1][1], csc_mat[1][2],
                                          csc_mat[2][0], csc_mat[2][1], csc_mat[2][2],
                                          i_bias_mat[0], i_bias_mat[1], i_bias_mat[2],
                                          o_bias_mat[0], o_bias_mat[1], o_bias_mat[2])
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set aipp color space convert parameters failed, return {ret}.")

    @property
    def swap_switch(self):
        return self._swap_switch

    @swap_switch.setter
    def swap_switch(self, swap_switch):
        """ Set aipp swap R/B or U/V swith flag.

        Args:
            swap_switch :swap swith

        Returns:
            None.
        """
        swap_switch = int(bool(swap_switch))
        ret = acl.mdl.set_aipp_rbuv_swap_switch(self.aipp_set, swap_switch)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set aipp swap R/B or U/V swith failed, return {ret}.")

        self._swap_switch = swap_switch

    @property
    def swap_ax(self):
        return self._swap_ax_switch

    @swap_ax.setter
    def swap_ax(self, swap_ax):
        """ Set aipp swap ax swith flag.

        Args:
            swap_ax :swap ax swith

        Returns:
            None.
        """
        swap_ax_switch = int(bool(swap_ax))
        ret = acl.mdl.set_aipp_ax_swap_switch(self.aipp_set, swap_ax_switch)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set aipp swap R/B or U/V swith failed, return {ret}.")

        self._swap_ax_switch = swap_ax_switch

    def set_means(self, means, input_id=0):
        """ Set aipp color channel data mean.

        Args:
            means : input mean value of channel 

        Returns:
            None.
        """
        if not isinstance(means, np.ndarray):
            raise TypeError(
                f"Input means expects a np.ndrray, but got {type(means)}.")

        assert means.shape[-1] == 3 or means.shape[-1] == 4, \
            f"Input means expects a 3x1 or 4x1 matrix, but got {means.shape} matrix."

        # means = [mean_chn0, mean_chn1, mean_chn2, mean_chn3]
        if means.shape[-1] == 3:
            means = np.append(means, np.array([0], dtype=means.dtype))

        ret = acl.mdl.set_aipp_dtc_pixel_mean(
            self.aipp_set, means[0], means[1], means[2], means[3], input_id)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set input mean value of channel failed, return {ret}.")

    def set_min(self, minm, input_id=0):
        """ Set aipp minimum pixel value.

        Args:
            minm : aipp pixel mean value  

        Returns:
            None.
        """
        if not isinstance(minm, np.ndarray):
            raise TypeError(
                f"Input minm expects a np.ndrray, but got {type(minm)}.")

        assert minm.shape[-1] == 3 or minm.shape[-1] == 4, \
            f"Input minm expects a 3x1 or 4x1 matrix, but got {minm.shape} matrix."

        # minm = [min_chn0, min_chn1, min_chn2, min_chn3]
        if minm.shape[-1] == 3:
            minm = np.append(minm, np.array([0], dtype=minm.dtype))

        ret = acl.mdl.set_aipp_dtc_pixel_min(
            self.aipp_set, minm[0], minm[1], minm[2], minm[3], input_id)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set input minimum value of channel failed, return {ret}.")

    def set_var(self, var, input_id=0):
        """ Set aipp variance of image.

        Args:
            var : aipp pixel variance value  

        Returns:
            None.
        """
        if not isinstance(var, np.ndarray):
            raise TypeError(
                f"Input var expects a np.ndrray, but got {type(var)}.")

        assert var.shape[-1] == 3 or var.shape[-1] == 4, \
            f"Input minm expects a 3x1 or 4x1 matrix, but got {var.shape} matrix."

        # var = [var_chn0, var_chn1, var_chn2, var_chn3]
        if var.shape[-1] == 3:
            var = np.append(var, np.array([1.0], dtype=var.dtype))

        ret = acl.mdl.set_aipp_pixel_var_reci(
            self.aipp_set, var[0], var[1], var[2], var[3], input_id)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set aipp swap aipp variance pixel value failed, return {ret}.")

    def set_crop(self, rect, input_id=0):
        """ Set aipp crop prarameters.

        Args:
            value : aipp pixel crop prarameters. 

        Returns:
            None.
        """
        if not isinstance(rect, np.ndarray):
            raise TypeError(
                f"Input var expects a np.ndrray, but got {type(rect)}.")

        assert rect.shape[-1] == 4, \
            f"Input rect expects a 4x1 matrix, but got {rect.shape} matrix."

        # caculate crop rectangle
        clipped_bbox = bbox_clip(rect, self.shape)
        startx, starty, w, h = clipped_bbox.tolist()

        ret = acl.mdl.set_aipp_crop_params(
            self.aipp_set, 1, startx, starty, w, h, input_id)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"set aipp swap aipp variance pixel value failed, return {ret}.")

    def set_aipp(self, model_id, dataset, input_id=0):
        """ Add aipp to model according to input model index.

        Args:
            model_id : input model id. 
            input_id : input io id
            dataset  : input dataset

        Returns:
            None.
        """
        ret = acl.mdl.set_aipp_by_input_index(
            model_id, dataset, input_id, self.aipp_set)
        if ret != ACL_SUCCESS:
            raise ValueError(
                f"Add aipp to model according to input model index failed, return {ret}.")

    def __del__(self):
        if hasattr(self, 'aipp_set'):
            ret = acl.mdl.destroy_aipp(self.aipp_set)
            assert ret == ACL_SUCCESS, f"destroy aipp {self.aipp_set} failed, return {ret}."
