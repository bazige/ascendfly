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
import os.path as osp
from pathlib import Path


def is_str(x):
    """judge x is a str type object

    Args:
        x : input x

    Returns:
        [bool]: 
        - True for x is str, 
        - False for x is not str
    """    
    return isinstance(x, str)


def is_filepath(x):
    """judge x is a filepath

    Args:
        x : input x

    Returns:
        [bool]: 
        - True: for x is filepath, 
        - False: for x is not filepath
    """    
    return is_str(x) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError('`filepath` should be a string or a Path')


def check_file_exist(filename, msg_tmpl="file '{}' does not exist"):
    """Check specific file is exist

    Args:
        filename (str): Input file name 
        msg_tmpl (str, optional): raise message. Defaults to "file '{}' does not exist".
    """    
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    """Make a directory

    Args:
        dir_name (str): The directory name
        mode (number, optional): The directory's permission. Defaults to 0o777.
    """    
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False):
    """AI is creating summary for scandir

    Args:
        dir_path (`Path` or str): Path of the directory.
        suffix (str or tuple(str), optional): File suffix that we are interested in. Defaults to None.
        recursive (bool, optional): If set to True, recursively scan the directory. Defaults to False.

    Returns:
        generator: A generator for all the interested files with relative pathes.

    Yields:
        rel_path: related path.
    """ 
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def find_vcs_root(path, markers=('.git', )):
    """Finds the root directory (including itself) of specified markers.

    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.

    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None


def traverse_file_paths(path, extensions, exclude_extensions=None):
    """Recursively reads all files under given folder, until all files have been ergodic.

    Args:
        path (`Path` or str): Path of the directory.
        extensions (str): File suffix.
        exclude_extensions (list, optional): Exclude extensions. Default: None.

    Returns:
        list: path_list contains all wanted files.
    """
    def is_valid_file(x):
        if exclude_extensions is None:
            return x.lower().endswith(extensions)
        else:
            return x.lower().endswith(extensions) and not x.lower().endswith(exclude_extensions)

    # check_file_exist(path)
    if isinstance(extensions, list):
        extensions = tuple(extensions)
    if isinstance(exclude_extensions, list):
        exclude_extensions = tuple(exclude_extensions)

    all_list = os.listdir(path)
    path_list = []
    for subpath in all_list:
        path_next = os.path.join(path, subpath)
        if os.path.isdir(path_next):
            path_list.extend(traverse_file_paths(
                path_next, extensions, exclude_extensions))
        else:
            if is_valid_file(path_next):
                path_list.append(path_next)
    return path_list
