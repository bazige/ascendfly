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
import os
import argparse


class CommandParser():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args_dict = {}
        self.init_arguement()


    def init_arguement(self):
        self.parser.add_argument('--device', nargs='+', type=int, default=[0])
        self.parser.add_argument('--model_path', type=str, default="./data/model/yolov3_aipp.om")
        self.parser.add_argument('--batch', type=int, default=1)
        self.parser.add_argument('--model_input_width', type=int, default=416)
        self.parser.add_argument('--model_input_height', type=int, default=416)
        self.parser.add_argument('--conf_path', type=str, default="./conf/cfg.ini")


    def command_parser(self):
        args = self.parser.parse_args()
        self.args_dict.update({'device': args.device})
        self.args_dict.update({'model_path': args.model_path})
        self.args_dict.update({'batch': args.batch})
        self.args_dict.update({'model_input_width': args.model_input_width})
        self.args_dict.update({'model_input_height': args.model_input_height})
        self.args_dict.update({'conf_path': args.conf_path})
        return self.args_dict
