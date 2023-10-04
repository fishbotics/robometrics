from typing import Any, Dict

import numpy as np
import yaml

from robometrics.file_structure import get_dataset_path

try:
    import geometrout
except ImportError:
    _HAS_GEOMETROUT = False
else:
    _HAS_GEOMETROUT = True


def structure_problems(problem_dict):
    assert (
        _HAS_GEOMETROUT
    ), "Optional package geometrout not installed, so this function is disabled"

    assert set(problem_dict.keys()) == set(
        [
            "collision_buffer_ik",
            "goal_ik",
            "goal_pose",
            "obstacles",
            "start",
            "world_frame",
        ]
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
            pos=np.array(problem_dict["goal_pose"]["position_xyz"]),
            quaternion=np.array(problem_dict["goal_pose"]["quaternion_wxyz"]),
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


def demo() -> Dict[str, Any]:
    raw_data = demo_raw()
    return structure_dataset(raw_data)
