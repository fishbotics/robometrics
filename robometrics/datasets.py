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


from typing import Any, Dict

import yaml

from robometrics.file_structure import get_dataset_path


def demo() -> Dict[str, Any]:
    path = get_dataset_path() / "demo_set.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def motion_benchmaker() -> Dict[str, Any]:
    path = get_dataset_path() / "mb_set.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def mpinets() -> Dict[str, Any]:
    path = get_dataset_path() / "mpinets_set.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
