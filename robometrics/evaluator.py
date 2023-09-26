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
from typing import Dict, List, Optional

from geometrout import SE3

from robometrics import metrics as rms
from robometrics.robometrics_types import EefTrajectory, Obstacles, Trajectory
from robometrics.robot import Robot


class Evaluator:
    """
    This class can be used to evaluate a whole set of environments and data
    """

    def __init__(self, robot: Robot):
        """
        Initializes the evaluator class

        """
        self.robot = robot
        self.groups: Dict[str, List[rms.TrajectoryMetrics]] = {}
        self.current_group: List[rms.TrajectoryMetrics] = []
        self.current_group_key: str = ""

    def create_new_group(self, key: str):
        """
        Creates a new metric group (for a new setting, for example)

        :param key str: The key for this metric group
        """
        self.groups[key] = []
        self.current_group_key = key
        self.current_group = self.groups[key]

    def evaluate_trajectory(
        self,
        trajectory: Trajectory,
        eef_trajectory: EefTrajectory,
        target_pose: SE3,
        obstacles: Obstacles,
        time: float,
        skip_metrics: bool = False,
    ) -> rms.TrajectoryMetrics:
        """
        Evaluates a single trajectory, stores the metrics in the current group and returns metrics.

        :param trajectory Trajectory: The trajectory
        :param eef_trajectory EefTrajectory: The trajectory followed by the Eef
        :param target_pose SE3: The target pose
        :param obstacles Obstacles: The obstacles in the scene
        :param time float: The time taken to calculate the trajectory
        :param skip_metrics bool: Whether to skip the path metrics (for example if it's
                                  a feasibility planner that failed)
        :rtype rms.TrajectoryMetrics: The metrics
        """
        if skip_metrics:
            metrics = rms.TrajectoryMetrics()
            self.current_group.append(metrics)
            return metrics
        (
            eef_pos_path,
            eef_orien_path,
        ) = rms.calculate_eef_path_lengths(eef_trajectory)
        metrics = rms.TrajectoryMetrics(
            skip=False,
            solve_time=time,
            collision=rms.in_collision(self.robot, trajectory, obstacles),
            joint_limit_violation=rms.violates_joint_limits(self.robot, trajectory),
            self_collision=rms.has_self_collision(self.robot, trajectory),
            physical_violation=rms.has_physical_violation(
                self.robot, trajectory, obstacles
            ),
            position_error=rms.position_error_in_cm(eef_trajectory[-1], target_pose),
            orientation_error=rms.orientation_error_in_degrees(
                eef_trajectory[-1].so3, target_pose.so3
            ),
            eef_position_path_length=eef_pos_path,
            eef_orientation_path_length=eef_orien_path,
            trajectory_length=len(trajectory),
        )
        metrics.success = metrics.position_error < 1 and not metrics.physical_violation
        self.current_group.append(metrics)
        return metrics

    @classmethod
    def metrics(cls, group: List[rms.TrajectoryMetrics]) -> rms.TrajectoryGroupMetrics:
        """
        Calculates the metrics for a specific group

        :param group Dict[str, Any]: The group of results
        :rtype Dict[str, float]: The metrics
        """
        return rms.TrajectoryGroupMetrics.from_list(group)

    @staticmethod
    def print_metrics(group: List[rms.TrajectoryMetrics]):
        """
        Prints the metrics in an easy to read format

        :param group Dict[str, float]: The group of results
        """
        metrics = rms.TrajectoryGroupMetrics.from_list(group)
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
