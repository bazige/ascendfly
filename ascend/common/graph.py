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
import objgraph


def show_growth(show_cnt=0):
    """Show python object references growth, the default show_cnt is 0.

    Args:
        show_cnt (int, optional): Debug memory release for show_cnt = 2. Defaults to 0.

    Typical usage example:
    ```python
    ascend.show_graph()
    ```
    """    
    print("------------------------")
    objgraph.show_growth()
    type_list = ['list', 'tuple', 'dict', 'method', 'weakref']
    print_str = "objgraph.by_type:\n"

    for by_type in type_list:
        type_len = len(objgraph.by_type(by_type))
        start = type_len - 1
        end = start - show_cnt
        print_str += f"{by_type} = {objgraph.by_type(by_type)[start:end:-1]}\n"
    print(print_str)
