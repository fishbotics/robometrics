# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os
import yaml


def get_module_path():
    path = os.path.dirname(__file__)
    return path


def get_content_path():
    root_path = get_module_path()
    path = os.path.join(root_path, "content")
    return path


def get_dataset_path():
    path = os.path.join(get_content_path(), "dataset")
    return path


def get_robot_path():
    path = os.path.join(get_content_path(), "robot")
    return path


def get_mpinets_dataset_path():
    path = os.path.join(get_dataset_path(), "mpinets_set.yaml")
    return path


def get_mb_dataset_path():
    path = os.path.join(get_dataset_path(), "mb_set.yaml")
    return path


def load_yaml(file_path: str):
    with open(file_path) as file:
        yaml_params = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_params


def get_mb_dataset():
    path = get_mb_dataset_path()
    return load_yaml(path)


def get_mpinets_dataset():
    path = get_mpinets_dataset_path()
    return load_yaml(path)


def join_path(str1, str2):
    return os.path.join(str1, str2)
