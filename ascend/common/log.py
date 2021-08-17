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
import logging
import sys
import os
from .path import mkdir_or_exist
from datetime import datetime
from .const import DEBUG, INFO, WARNING, ERROR


level_dict = {
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARNING: logging.WARNING,
    ERROR: logging.ERROR
}


class Log():
    """Define a Log object, to print log info to file.

    Attributes:
        level (const): Log level, one of [`DEBUG`, `INFO`, `WARNING`, `ERROR`]
        where (str):  Logging place configure func, one of ['file', 'console', 'both']

    Typical usage example:
    ```python
    ascend.Log(ERROR, 'Input parameters invalid.')
    ```
    """

    def __init__(self):
        real_path = os.getcwd() + '/log/'
        mkdir_or_exist(real_path)
        now_time = datetime.now().strftime('%Y-%m-%d')
        file_name = 'AscendFly-' + now_time+'.log'
        self.formatter = logging.Formatter('[%(levelname)s] %(funcName)s:%(asctime)s '
                                           + '[%(filename)s:%(lineno)s] messages:%(message)s')

        self.logger = logging.getLogger('demo_log')
        self.cnsl_handler = logging.StreamHandler(sys.stdout)
        self.file_handler = logging.FileHandler(real_path + file_name)
        self.cnsl_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)
        self.level = INFO
        self.where = 'file'

    @property
    def level(self):
        if hasattr(self, '_level'):
            return self._level

    @level.setter
    def level(self, level=ERROR):
        """Define a level configure func to set log level.

        Args:
            level : one of [DEBUG, INFO, WARNING, ERROR]

        Returns:
            None
        """
        try:
            self._level = level_dict[level]
        except KeyError:
            raise ValueError(f"Input level is invalid.")
        self.cnsl_handler.setLevel(level=self._level)
        self.file_handler.setLevel(level=self._level)
        self.logger.setLevel(level=self._level)

    @property
    def where(self):
        if hasattr(self, '_place'):
            return self._place

    @where.setter
    def where(self, place='file'):
        """Define a logging place configure func.

        Args:
            place : one of ['file', 'console', 'both']

        Returns:
            None
        """
        if place not in ['file', 'console', 'both']:
            raise ValueError(
                f"Input 'place' shoule be one of 'file', 'console', 'both'.")

        if place in ['file', 'both']:
            self.logger.addHandler(self.file_handler)
        if place in ['console', 'both']:
            self.logger.addHandler(self.cnsl_handler)
        self._place = place

    def __call__(self, level, message):
        if level == DEBUG:
            self.logger.debug(message)
        elif level == INFO:
            self.logger.info(message)
        elif level == WARNING:
            self.logger.warning(message)
        elif level == ERROR:
            self.logger.error(message)


# reassign to other name
Log = Log()
