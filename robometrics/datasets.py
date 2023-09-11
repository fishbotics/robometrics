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

import numpy as np
import yaml

from robometrics.file_structure import get_dataset_path


def structure_problems(problem_dict):
    try:
        import geometrout
    except ImportError:
        raise ImportError(
            "Optional package geometrout not installed, so this function is disabled"
        )

    assert set(problem_dict.keys()) == set(
        ["collision_buffer_ik", "goal_ik", "goal_pose", "obstacles", "start"]
    )
    assert set(problem_dict["obstacles"].keys()) == set(["cuboid", "cylinder"])
    obstacles = {
        "cylinder": {
            k: geometrout.Cylinder(
                height=v["height"],
                radius=v["radius"],
                center=np.array(v["pose"][:3]),
                quaternion=np.array(v["pose"][3:]),
            )
            for k, v in problem_dict["obstacles"]["cylinder"].items()
        },
        "cuboid": {
            k: geometrout.Cuboid(
                dims=np.array(v["dims"]),
                center=np.array(v["pose"][:3]),
                quaternion=np.array(v["pose"][3:]),
            )
            for k, v in problem_dict["obstacles"]["cuboid"].items()
        },
    }
    return {
        "collision_buffer_ik": problem_dict["collision_buffer_ik"],
        "goal_ik": [np.asarray(ik) for ik in problem_dict["goal_ik"]],
        "goal_pose": geometrout.SE3(
            pos=np.array(problem_dict["goal_pose"][:3]),
            quaternion=np.array(problem_dict["goal_pose"][3:]),
        ),
        "obstacles": obstacles,
    }


def demo_raw() -> Dict[str, Any]:
    path = get_dataset_path() / "demo_set.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def structure_dataset(raw_dataset: Dict[str, Any]) -> Dict[str, Any]:
    for problem_set in raw_dataset.values():
        for ii, v in enumerate(problem_set):
            problem_set[ii] = structure_problems(v)
    return raw_dataset


def motion_benchmaker_raw() -> Dict[str, Any]:
    path = get_dataset_path() / "mb_set.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def motion_benchmaker() -> Dict[str, Any]:
    raw_data = motion_benchmaker_raw()
    return structure_dataset(raw_data)


def mpinets_raw() -> Dict[str, Any]:
    path = get_dataset_path() / "mpinets_set.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def mpinets() -> Dict[str, Any]:
    raw_data = mpinets_raw()
    return structure_dataset(raw_data)
