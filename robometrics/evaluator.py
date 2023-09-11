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

import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from geometrout import SE3, SO3, Cuboid, Cylinder, Sphere
except ImportError:
    raise ImportError(
        "Optional package geometrout not installed, so this function is disabled"
    )

from robometrics.metrics import TrajectoryGroupMetrics, TrajectoryMetrics
from robometrics.robot import Robot

Obstacles = List[Union[Cuboid, Cylinder, Sphere]]
Trajectory = Sequence[Union[Sequence, np.ndarray]]
EefTrajectory = Sequence[SE3]


def check_final_position(final_pose: SE3, target: SE3) -> float:
    """
    Gets the number of centimeters between the final pose and target

    :param final_pose SE3: The final pose of the trajectory
    :param target SE3: The target pose
    :rtype float: The distance in centimeters
    """
    return 100 * np.linalg.norm(final_pose.pos - target.pos)


def check_final_orientation(final_orientation: SO3, target: SO3) -> float:
    """
    Gets the number of degrees between the final orientation and the target
    orientation

    :param final_orientation SO3: The final orientation
    :param target SO3: The final target orientation
    :rtype float: The rotational distance in degrees
    """
    return np.abs((final_orientation * target.conjugate).degrees)


class Evaluator:
    """
    This class can be used to evaluate a whole set of environments and data
    """

    def __init__(self, robot: Robot):
        """
        Initializes the evaluator class

        """
        self.robot = robot
        self.groups: Dict[str, List[TrajectoryMetrics]] = {}
        self.current_group: List[TrajectoryMetrics] = []
        self.current_group_key: str = ""

    def create_new_group(self, key: str):
        """
        Creates a new metric group (for a new setting, for example)

        :param key str: The key for this metric group
        """
        self.groups[key] = []
        self.current_group_key = key
        self.current_group = self.groups[key]

    def has_self_collision(
        self,
        trajectory: Trajectory,
    ) -> bool:
        """
        Checks whether there is a self collision (using OR between different methods)

        :param trajectory Trajectory: The trajectory
        :rtype bool: Whether there is a self collision
        """
        for q in trajectory:
            if self.robot.has_self_collision(q):
                return True
        return False

    def in_collision(
        self,
        trajectory: Trajectory,
        obstacles: Obstacles,
    ) -> bool:
        """
        Checks whether the trajectory is in collision according to all including
        collision checkers (using AND between different methods)

        :param trajectory Trajectory: The trajectory
        :param obstacles Obstacles: Obstacles to check
        :rtype bool: Whether there is a collision
        """
        for q in trajectory:
            if self.robot.has_scene_collision(q, obstacles):
                return True
        return False

    def violates_joint_limits(self, trajectory: Trajectory) -> bool:
        """
        Checks whether any configuration in the trajectory violates joint limits

        :param trajectory Trajectory: The trajectory
        :rtype bool: Whether there is a joint limit violation
        """
        for q in trajectory:
            if not self.robot.within_joint_limits(q):
                return True
        return False

    def has_physical_violation(
        self, trajectory: Trajectory, obstacles: Obstacles
    ) -> bool:
        """
        Checks whether there is any physical violation
        Checks collision, self collision, joint limit violation

        :param trajectory Trajectory: The trajectory
        :param obstacles Obstacles: The obstacles in the scene
        :rtype bool: Whether there is at least one physical violation
        """
        return (
            self.in_collision(trajectory, obstacles)
            or self.violates_joint_limits(trajectory)
            or self.has_self_collision(trajectory)
        )

    def calculate_eef_path_lengths(
        self, eef_trajectory: EefTrajectory
    ) -> Tuple[float, float]:
        """
        Calculate the end effector path lengths (position and orientation).
        Orientation is in degrees.

        :param trajectory Trajectory: The trajectory
        :rtype Tuple[float, float]: The path lengths (position, orientation)
        """

        eef_positions = np.asarray([pose.pos for pose in eef_trajectory])
        assert eef_positions.ndim == 2 and eef_positions.shape[1] == 3
        position_step_lengths = np.linalg.norm(
            np.diff(eef_positions, 1, axis=0), axis=1
        )
        eef_position_path_length = sum(position_step_lengths)

        eef_so3 = [pose.so3 for pose in eef_trajectory]
        eef_orientation_path_length = 0
        for qi, qj in zip(eef_so3[:-1], eef_so3[1:]):
            eef_orientation_path_length += np.abs((qj * qi.conjugate).degrees)
        return eef_position_path_length, eef_orientation_path_length

    def evaluate_trajectory(
        self,
        trajectory: Trajectory,
        eef_trajectory: EefTrajectory,
        target_pose: SE3,
        obstacles: Obstacles,
        time: float,
        skip_metrics: bool = False,
    ):
        """
        Evaluates a single trajectory and stores the metrics in the current group.

        :param trajectory Trajectory: The trajectory
        :param eef_trajectory EefTrajectory: The trajectory followed by the Eef
        :param target_pose SE3: The target pose
        :param obstacles Obstacles: The obstacles in the scene
        :param time float: The time taken to calculate the trajectory
        :param skip_metrics bool: Whether to skip the path metrics (for example if it's
                                  a feasibility planner that failed)
        """
        if skip_metrics:
            self.current_group.append(TrajectoryMetrics())
            return
        (
            eef_pos_path,
            eef_orien_path,
        ) = self.calculate_eef_path_lengths(eef_trajectory)
        metrics = TrajectoryMetrics(
            skip=False,
            time=time,
            collision=self.in_collision(trajectory, obstacles),
            joint_limit_violation=self.violates_joint_limits(trajectory),
            self_collision=self.has_self_collision(trajectory),
            physical_violation=self.has_physical_violation(trajectory, obstacles),
            position_error=check_final_position(eef_trajectory[-1], target_pose),
            orientation_error=check_final_orientation(
                eef_trajectory[-1].so3, target_pose.so3
            ),
            eef_position_path_length=eef_pos_path,
            eef_orientation_path_length=eef_orien_path,
            trajectory_length=len(trajectory),
        )
        metrics.success = metrics.position_error < 1 and not metrics.physical_violation
        self.current_group.append(metrics)

    @classmethod
    def metrics(cls, group: List[TrajectoryMetrics]) -> TrajectoryGroupMetrics:
        """
        Calculates the metrics for a specific group

        :param group Dict[str, Any]: The group of results
        :rtype Dict[str, float]: The metrics
        """
        return TrajectoryGroupMetrics.from_list(group)

    @staticmethod
    def print_metrics(group: List[TrajectoryMetrics]):
        """
        Prints the metrics in an easy to read format

        :param group Dict[str, float]: The group of results
        """
        metrics = TrajectoryGroupMetrics.from_list(group)
        metrics.print_summary()

    def save_group(self, directory: str, test_name: str, key: Optional[str] = None):
        """
        Save the results of a single group

        :param directory str: The directory in which to save the results
        :param test_name str: The name of this specific test
        :param key Optional[str]: The group key to use. If not specified will use the
                                  current group
        """
        if key is None:
            group = self.current_group
        else:
            group = self.groups[key]
        save_path = Path(directory) / f"{test_name}_{self.current_group_key}.pkl"
        print(f"Saving group metrics to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(group, f)

    def save(self, directory: str, test_name: str):
        """
        Save all the groups

        :param directory str: The directory name in which to save results
        :param test_name str: The test name (used as the file name)
        """
        save_path = Path(directory) / f"{test_name}_metrics.pkl"
        print(f"Metrics will save to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(self.groups, f)

    def print_group_metrics(self, key: Optional[str] = None):
        """
        Prints out the metrics for a specific group

        :param key Optional[str]: The group key (if none specified, will use current
                                  group)
        """
        if key is not None:
            self.current_group = self.groups[key]
            self.current_group_key = key
        return self.print_metrics(self.current_group)

    def print_overall_metrics(self):
        """
        Prints the metrics for the aggregated results over all groups
        """
        supergroup = list(itertools.chain.from_iterable(self.groups.values()))
        return self.print_metrics(supergroup)
