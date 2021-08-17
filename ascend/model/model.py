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
import weakref

from collections import OrderedDict
from ..common.const import *
from ..common.log import Log
from ..resource.mem import memcpy_d2d
from ..resource.context import create_stream
from ..data.ascendarray import AscendArray

class DataSet():
    """Define a DataSet class to manage input/output buffer of model.

    Attributes:
        dataset (dataset): The created dataset

    """    
    def __init__(self):
        self._class_name = self.__class__.__name__
        self._dataset = acl.mdl.create_dataset()

    @property
    def dataset(self):
        return self._dataset

    def add_buffer(self, ascendarray):
        """Bind an AscendArray to the data buffer, and then add 
            the data buffer to the created dataset.

        Args:
            ascendarray (AascendArray): Input data to be added in the model buffer.

        Raises:
            ValueError: add failed with acl.mdl.add_dataset_buffer api.
        """        
        array = weakref.ref(ascendarray)
        assert isinstance(array(), AscendArray), \
                f"Input args of func 'add_buffer' expects a class of AscendArray, but got {type(array())}."

        # add data buffer to the dataset
        buffer = acl.create_data_buffer(array().ascend_data, array().nbytes)
        _, ret = acl.mdl.add_dataset_buffer(self._dataset, buffer)
        if ret != ACL_SUCCESS:
            acl.destroy_data_buffer(buffer)
            del array
            raise ValueError('Return value of add_buffer should be ACL_SUCCESS(0). '
                f'But received {ret}')

    def __del__(self):
        if hasattr(self, '_dataset'):
            num_buffers = acl.mdl.get_dataset_num_buffers(self._dataset)
            for idx in range(num_buffers):
                data_buf = acl.mdl.get_dataset_buffer(self._dataset, idx)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    assert (ret == ACL_SUCCESS), f"Fail to destroy data buffer, return {ret}."

            ret = acl.mdl.destroy_dataset(self._dataset)
            if ret != ACL_SUCCESS:
                raise ValueError(f"Failed to destroy output dataset, return {ret}.")

class ModelDesc():
    """Define a ModelDesc class to manage loaded model.

    Attributes:
        desc (int): Created model desc
    """  
    def __init__(self, model_id):
        self._class_name = self.__class__.__name__
        self._model_desc = acl.mdl.create_desc()

        ret = acl.mdl.get_desc(self._model_desc, model_id)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to init model desc in class ModelDesc \
                    because of get_desc return {ret}.")
        
    @property
    def desc(self):
        return self._model_desc

    def __del__(self):
        ret = acl.mdl.destroy_desc(self._model_desc)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Fail to destroy model desc, return {ret}.")


class AscendModel():
    """Define a AscendModel class like to manage model and inferring.

    Attributes:
        context (int): Context resource of this model working on.
        stream (int): Stream resource of this model working on.
        model_id (int): This model's id
        tensor (dict): A ordered-dict to save all input and output tensors
        net_in_n (int): Model's input numbers
        net_out_n (int): Model's output numbers
        dataset_in (DataSet obj): Input dataset object of model
        dataset_out (DataSet obj): Output dataset object of model
        tensor_names (str): Model's input and output tensors' name.

    .. hint:: 
        - run                : do model inference
        - feed_data          : feed AscendArray data to model's input port
        - get_tensor_by_name : get the output result of inference according to output tensor name
        - set_batch          : Set model input dynamic batch size.
        - set_shape          : Set model input dynamic shape.

    Typical usage example:
    ```python
    model = ascend.AscendModel(context, model_path='./yolov3.om')
    print(model.tensor_names)
    ```
    """
    def __init__(self,
                context,
                model_path):
        self.class_name = self.__class__.__name__
        self.context = context
        self.stream = create_stream(context)

        self.tensor = OrderedDict()
        self.net_in_n = 0
        self.net_out_n = 0
        self.dataset_in = None
        self.dataset_out = None
        self._tensor_names = {'input':[], 'output':[]}

        # load model and initial
        self.__load_model(model_path)

    def __load_model(self, model_path:str):
        """load model and create input and output tensor.
        Args:
            model_path.

        Returns:
            None.
        """
        if not isinstance(model_path, str):
            raise TypeError('model_path must be a string. '
                        f'But received {type(model_path)}')

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret != ACL_SUCCESS:
            raise ValueError(f'Failed to load model from file {model_path}, return {ret}.')

        # creating the model instance, which only be created once.
        self.__model_desc = ModelDesc(self.model_id)

        self.net_in_n = acl.mdl.get_num_inputs(self.__model_desc.desc)
        if self.net_in_n <= 0:
            raise ValueError('input num of model should larger than 0. '
                        f'But received {self.net_in_n}')

        self.net_out_n = acl.mdl.get_num_outputs(self.__model_desc.desc)
        if self.net_out_n <= 0:
            raise ValueError('output num of model should larger than 0. '
                        f'But received {self.net_out_n}')

        self.__create_input_tensor(self.__model_desc.desc, self.net_in_n)
        Log(INFO, 'create input tensor success.')

        self.__create_output_tensor(self.__model_desc.desc, self.net_out_n)
        Log(INFO, 'create output tensor success.')

    def __create_input_tensor(self, model_desc, net_input_num:int):
        """create input tensor and band memory pointer.
        Args:
            model_desc: model's information and description.
            net_input_num: input node number of the net.
        Returns:
            None.
        """
        if model_desc is None:
            raise ValueError('Input param model_desc should not be None.')
        if not isinstance(net_input_num, int):
            raise TypeError('Input param net_input_num must be an int. '
                    f'But received {type(net_input_num)}')
        if net_input_num <= 0:
            raise ValueError('Input param net_input_num should larger than 0. '
                    f'But received {net_input_num}')

        self.dataset_in = DataSet()
        Log(INFO, ' create input dataset success.')

        for idx in range(net_input_num):
            dims, ret = acl.mdl.get_input_dims_v2(model_desc, idx)
            if ret != ACL_SUCCESS:
                raise ValueError(f'Failed to get input dims at {idx} in func \
                        __create_input_tensor. Return value {ret}.')

            # get input acl tensor name and data type.
            tensor_name = dims["name"]
            tensor_dtype = acl.mdl.get_input_data_type(model_desc, idx)
            tensor_format = acl.mdl.get_input_format(model_desc, idx)
            tensor_format = tensor_fmt_map[tensor_format]

            # create a tensor of class AscendArray and assign attributes to it.
            self.tensor[tensor_name] = AscendArray(shape=tuple(dims["dims"]), dtype=dtype_dict[tensor_dtype], \
                                                    format=tensor_format)

            # banding the dataset with data buffer
            self.dataset_in.add_buffer(self.tensor[tensor_name])

            # save tensor name to input
            self._tensor_names['input'].append(tensor_name)

        Log(INFO, 'create input tensor success.')

    def __create_output_tensor(self, model_desc, net_output_num):
        """create output tensor and band memory pointer.
        Args:
            model_desc: model's information and description.
            net_output_num: output tensor number of the net.
        Returns:
            None.
        """
        if model_desc is None:
            raise ValueError('Input param model_desc should not be None.')
        if not isinstance(net_output_num, int):
            raise TypeError('Input param net_output_num must be an int. '
                    f'But received {type(net_output_num)}')
        if net_output_num <= 0:
            raise ValueError('Input param net_output_num should larger than 0. '
                    f'But received {net_output_num}')

        self.dataset_out = DataSet()
        Log(INFO, ' create output dataset success.')

        for idx in range(net_output_num):
            dims, ret = acl.mdl.get_output_dims(model_desc, idx)
            if ret != ACL_SUCCESS:
                raise ValueError(f'Failed to get output dims at {idx} in func \
                        __create_output_tensor. Return value {ret}.')

            # get the acl tensor name and data type.
            tensor_name = dims["name"]
            tensor_dtype = acl.mdl.get_output_data_type(model_desc, idx)
            tensor_format = acl.mdl.get_output_format(model_desc, idx)
            tensor_format = tensor_fmt_map[tensor_format]
            
            # create a tensor of class AscendArray and assign attributes to it.
            self.tensor[tensor_name] = AscendArray(shape=tuple(dims["dims"]), dtype=dtype_dict[tensor_dtype], \
                                                    format=tensor_format)

            # banding the dataset with data buffer
            self.dataset_out.add_buffer(self.tensor[tensor_name])

            # save tensor name to output
            self._tensor_names['output'].append(tensor_name)

        Log(INFO, 'create output tensor success.')

    def __check_input(self, data, net_in):
        """ check input data is matched with model input
        Args:
            data   : input image or data.
            net_in : model input tensor

        Returns:
            None.
        """
        if not isinstance(data, AscendArray):
            raise TypeError(f"feed data expects an AscendArray, but got {type(data)}.")
            
        Log(INFO, f"data format: {data.format}, net_in format: {net_in.format}.")

        # case 1: input yuv420_nv12 or yuv420_nv21 or yuv400 image
        if data.ndim == 2:
            # only judge channel is same to model input
            if (net_in.format == 'NCHW' and net_in.shape[1] != 1) or \
                (net_in.format == "NHWC" and net_in.shape[3] != 1):
                raise ValueError(f"Input tensor's expects a single channel, but got {net_in.shape}.")

        # case 2: input RGB/BGR or 3-dims tensor
        if data.ndim == 3:
            # only judge channel is same to model input
            if (net_in.format == 'NCHW' and net_in.shape[1] != 3) or \
                (net_in.format == "NHWC" and net_in.shape[3] != 3):
                raise ValueError(f"Input tensor's expects a single channel, but got {net_in.shape}.")

        # case 3: input 4-dims tensor
        if data.ndim == 4:
            assert data.shape == net_in.shape, \
                f"Feed data expects same shape to model input {net_in.shape}, but got {data.shape}."

        # if elements of input data not equal to model input, print warnning log.
        if data.size != net_in.size:
            Log(WARNING, f"feed data size {data.shape} not suit to model input {net_in.shape}.")

    def run(self):
        """Run model offline inference.

        ```python
        # model is the instantiated AscendModel obj
        model.run()
        ```
        """
        ret = acl.mdl.execute_async(
                        self.model_id,
                        self.dataset_in.dataset,
                        self.dataset_out.dataset,
                        self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to excute async in model inference, return {ret}.")
        
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Failed to synchronize stream in model inference, return {ret}.")

        Log(INFO, 'model inference success.')


    def feed_data(self, args:dict):
        """Feed data to model.

        Args:
            args (dict): Feed specific tensor's data to the model.

        Typical usage example: 
        ```python
        # 'input_tensor*' is the node of model input, and input_data* is the preprocessed data(image).
        model.feed_data({'input_tensor1':input_data1, 'input_tensor2':input_data2})
        ```
        """
        if not isinstance(args, dict):
            raise TypeError(f'Input feed_data expects a dict, but received {type(args)}.')

        if len(args) <= 0:
            raise TypeError('Input args of feed_data is null.')

        if len(args) != self.net_in_n:
            raise ValueError('Input tensor number to model should be equal to the net('
                f'{self.net_in_n}). But the actual input is: {len(args)}')

        for key, in_tensor in args.items():
            assert (key in self.tensor), \
                f'Tensor name {key} is not the input or this tensor name is not correct,' \
                 + f' you should choose one of {self._tensor_names}.'

            # feed data process will separate to below steps:
            # step 1: check input data is matched with model input
            self.__check_input(in_tensor, self.tensor[key])

            # step 2:
            #   copy input tensor's data(the front module's output data) ptr to variable input_ptr, 
            #   while input_tensor_dict.values() is a class of AscendArray
            input_ptr = in_tensor.ascend_data

            # step 3:
            #   copy model input tensor's data ptr to variable tensor_ptr
            tensor_ptr = self.tensor[key].ascend_data

            # step 4:
            #   get data size that will be copy to model input dataset
            size = min(self.tensor[key].nbytes, in_tensor.nbytes)

            # step 5:
            #   do copy
            memcpy_d2d(tensor_ptr, input_ptr, size)

        Log(INFO, 'feed data to model success.')


    def get_tensor_by_name(self, out_node_name:str):
        """Get tensor data from output of model inference.
        Args:
            out_node_name (str): Get specific tensor's data by name. 

        Returns:
            AscendArray: Output tensor of node out_node_name after model inference.

        Typical usage example: 
        ```python
        # 'output_tensor_name1' and 'output_tensor_name2' is the node of model output
        output_tensor1 = model.get_tensor_by_name('output_tensor_name1')
        output_tensor2 = model.get_tensor_by_name('output_tensor_name2')
        ```
        """
        assert isinstance(out_node_name, str), \
                f'Func args of get_tensor_by_name expects str type, but got {type(out_node_name)}'
        
        assert (out_node_name in self.tensor), \
                f'Tensor name [{out_node_name}] is not the output tensor or this tensor name is not correct,' \
                 + f' you should choose one of {self._tensor_names}.'

        return self.tensor[out_node_name]
    
    @property
    def tensor_names(self):
        """Get input and output tensors' name from model.

        Returns:
            list: Input/output node's name of model.

        Typical usage example: 
        ```python
        #'tensor_names' is the node of model input and output.
        tensor_names = model.tensor_names
        ```
        """
        return self._tensor_names


    def set_batch(self, tensor_name, batch):
        """Set model input dynamic batch size.

        Args:
            tensor_name (str): Input tensor's name
            batch (int): Configurated tensor's batch size
        """
        if not isinstance(batch, int):
            raise ValueError(f"Input batch expects an int, but got {type(batch)}.")
        
        idx, ret = acl.mdl.get_input_index_by_name(self.__model_desc.desc, tensor_name)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Input tensor name {tensor_name} is not a input.")

        ret = acl.mdl.set_dynamic_batch_size(self.model_id, self.dataset_in, idx, batch)
        if ret != ACL_SUCCESS:
            raise ValueError(f"set dynamic batch size failed for input {idx}, return {ret}.")

    def set_shape(self, tensor_name, shape):
        """Set model input dynamic shape.
        Args:
            shape (tuple): a tuple (w, h) of dynamic input shape

        Returns:
            None.
        """
        if not isinstance(shape, tuple):
            raise ValueError(f"Input shape expects an tuple, but got {type(shape)}.")
        
        idx, ret = acl.mdl.get_input_index_by_name(self.__model_desc.desc, tensor_name)
        if ret != ACL_SUCCESS:
            raise ValueError(f"Input tensor name {tensor_name} is not a input.")

        ret = acl.mdl.set_dynamic_hw_size(self.model_id, self.dataset, idx, shape[1], shape[0])
        if ret != ACL_SUCCESS:
            raise ValueError(f"set dynamic shape failed for input {idx}, return {ret}.")

    def __release_model(self):
        """release data and unload model.
        Args:
            None

        Returns:
            None.
        """
        if hasattr(self, '__model_desc'):
            del self.__model_desc

        if hasattr(self, 'model_id') and self.model_id:
            ret = acl.mdl.unload(self.model_id)
            assert ret == ACL_SUCCESS, f'Unload model with id: {self.model_id} failded.'

        if hasattr(self, 'stream'):
            ret = acl.rt.destroy_stream(self.stream)
            assert ret == ACL_SUCCESS, \
                f"destroy stream error in func __release_model, return {ret}."

        Log(INFO, 'release model success')

    def __del__(self):
        ctx, ret = acl.rt.get_context()
        if ret != ACL_SUCCESS or ctx is None:
            raise ValueError("Release AscendModel instance failed, because context is not available.")
             
        del self.dataset_in
        del self.dataset_out
        self.__release_model()


