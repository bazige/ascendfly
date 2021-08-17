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
from configparser import ConfigParser
from .const import H265_MAIN_LEVEL,H264_BASELINE_LEVEL,H264_MAIN_LEVEL,H264_HIGH_LEVEL,ERROR 
from .log import Log


class ConfigParserModule():
    def __init__(self, config_file_path):
        self.cf = ConfigParser()
        try:
            if config_file_path is None:
                self.current_file_path = os.path.dirname(os.path.realpath(__file__))
                self.config_file_path = os.path.join(self.current_file_path, "cfg.ini")
            else:
                self.config_file_path = config_file_path
                self.cf.read(self.config_file_path, encoding="utf-8")
        except Exception as e:
            raise e

        self.item_dict = {}
        self.options_list = []
        self.section_dict = {}

    def get_item_as_dict_from_section(self, section_name):
        options_dict = dict(self.cf.items(section_name))
        return options_dict

    def get_options_from_section(self, section_name):
        options = self.cf.options(section_name)
        return options

    def get_option_value(self, section_name, option_name):
        value = self.cf.get(section_name, option_name)
        return value

    def get_section(self):
        return self.cf.sections()

    def remove_section(self, section):
        self.cf.remove_section(section)

    def remove_option(self, section, key):
        self.cf.remove_option(section, key)

    def add_section(self, section):
        self.cf.add_section(section)


class ConfigInfo(object):
    
    def __init__(self, config_path):
        self.class_name = self.__class__.__name__
        self.config = ConfigParserModule(config_path)
        self.config_dict = dict()


    def getSectionlist(self):
        sections = self.config.get_section()
        for section in sections:
            self.config_dict[section] = self.getOptions(section)
        return self.config_dict


    def getOptions(self, section):
        optionlist = list()
        options = self.config.get_options_from_section(section)
        for option in options:
            value = self.config.get_option_value(section, option)
            optiondict = dict()
            optiondict[option] = value
            optionlist.append(optiondict)
        return optionlist


    def getConfigInfo(self):
        config = dict()
        config_info_dict = self.getSectionlist()
        config['channel_count'] = int(config_info_dict['video_stream'][0]['channel_count'])
        channel_id_list = config_info_dict['video_stream'][1]['channel_id'].split(',')
        config['channel_id'] = list(map(lambda x: int(x), channel_id_list))
        config['rtsp'] = config_info_dict['video_stream'][2]['rtsp'].split(',')
        config['width'] = int(config_info_dict['video_stream'][3]['width'])
        config['height'] = int(config_info_dict['video_stream'][4]['height'])
        encode_type = config_info_dict['video_stream'][5]['encode_type']
        if encode_type == 'H264':
            config['encode_type'] =  H264_BASELINE_LEVEL
        elif encode_type == 'H265':
            config['encode_type'] =  H265_MAIN_LEVEL
        else:
            Log(ERROR, 'encode_type is not supported in getConfigInfo.')
        return config


if __name__ == '__main__':
    conInfo = ConfigInfo("/home/PyAcl_demo_test/PyAcl_demo/src/conf/cfg.ini")
    config_info_dict = conInfo.getConfigInfo()
    print(config_info_dict) 
