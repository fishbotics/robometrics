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
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from geometrout.primitive import Cuboid, Cylinder, Sphere
from geometrout.transform import SE3, SO3

from robometrics.robot import Robot

Stat = namedtuple("Stat", "mean std")

Obstacles = List[Union[Cuboid, Cylinder, Sphere]]
Trajectory = Sequence[Union[Sequence, np.ndarray]]
EefTrajectory = Sequence[SE3]


def make_stat(lst: Sequence[float]):
    return Stat(np.mean(lst), np.std(lst))


def correct_negative_volumes(
    volumes: List[Obstacles], target_pose: SE3
) -> List[Obstacles]:
    """
    Filters out any invalid volumes from a list of negative volumes. Invalid volumes
    would be ones in which the target pose is inside. This would make all trajectories
    to that final pose invalid (because a condition of success is that the final eef
    is not inside any of the negative volumes)

    :param volumes List[Obstacles]: A list of negative volumes
    :param target_pose SE3: The target pose for a trajectory
    :return List[Obstacles]: The valid negative volumes (i.e. ones where the target is not inside one)
    """
    return [v for v in volumes if v.sdf(target_pose.pos) > 0]


def percent_true(arr: Sequence) -> float:
    """
    Returns the percent true of a boolean sequence or the percent nonzero of a numerical
    sequence

    :param arr Sequence: The input sequence
    :rtype float: The percent
    """
    return 100 * np.count_nonzero(arr) / len(arr)


@dataclass
class TrajectoryMetrics:
    skip: bool = True
    success: bool = False
    time: float = np.inf
    collision: bool = True
    joint_limit_violation: bool = True
    self_collision: bool = True
    physical_violation: bool = True
    position_error: float = np.inf
    orientation_error: float = np.inf
    eef_position_path_length: float = np.inf
    eef_orientation_path_length: float = np.inf
    num_steps: int = -1


@dataclass
class TrajectoryGroupMetrics:
    group_size: int  # Num trajectories in group
    success: float  # Success % in group
    skips: int  # % in group with no solution
    time: Stat  # Mean/Std of solution time for unskipped problems
    step_time: Stat  # Mean/Std of time per action (not always useful)
    env_collision_rate: float  # % in group with environment collisions
    self_collision_rate: float  # % in group with self collisions
    joint_violation_rate: float  # % in group with a joint limit violation
    physical_violation_rate: float  # % in group with any physical violation
    within_one_cm_rate: float  # % in group that gets within 1cm of target
    within_five_cm_rate: float  # % in group that gets within 5cm of target
    within_fifteen_deg_rate: float  # % in group that gets within 15deg of target
    within_thirty_deg_rate: float  # % in group that gets within 30deg of target
    eef_position_path_length: Stat  # Mean/Std of Euclidean eef path length
    eef_orientation_path_length: Stat  # Mean/Std of orientation path length (deg)


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

    @staticmethod
    def check_final_position(final_pose: SE3, target: SE3) -> float:
        """
        Gets the number of centimeters between the final pose and target

        :param final_pose SE3: The final pose of the trajectory
        :param target SE3: The target pose
        :rtype float: The distance in centimeters
        """
        return 100 * np.linalg.norm(final_pose.pos - target.pos)

    @staticmethod
    def check_final_orientation(final_orientation: SO3, target: SO3) -> float:
        """
        Gets the number of degrees between the final orientation and the target
        orientation

        :param final_orientation SO3: The final orientation
        :param target SO3: The final target orientation
        :rtype float: The rotational distance in degrees
        """
        return np.abs((final_orientation * target.conjugate).degrees)

    @staticmethod
    def check_final_region(
        final_pose: SE3,
        target_volume: Union[Cuboid, Cylinder, Sphere],
        negative_volumes: Obstacles,
    ) -> bool:
        """
        Checks that the final pose is in the correct region if one is specified. For
        example, when reaching inside a drawer, it is not sufficient to be close to the
        final target as a condition of success. Instead, the final pose might be in
        the correct drawer (and likewise not in the incorrect drawer)

        :param final_pose SE3: The final pose of the trajectory
        :param target_volume Union[Cuboid, Cylinder, Sphere]: The volume to be inside of
        :param negative_volumes Obstacles: Volumes to be outside of
        :rtype bool: Whether the check is successful and the target is in the right
                     region and outside the wrong regions
        """
        return target_volume.sdf(final_pose.pos) <= 0 and np.all(
            [v.sdf(final_pose.pos) > 0 for v in negative_volumes]
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
        eef_trajectory: Trajectory,
        target_pose: SE3,
        obstacles: Obstacles,
        target_volume: Union[Cuboid, Cylinder, Sphere],
        target_negative_volumes: Obstacles,
        time: float,
        skip_metrics: bool = False,
    ):
        """
        Evaluates a single trajectory and stores the metrics in the current group.

        :param trajectory Trajectory: The trajectory
        :param eef_trajectory EefTrajectory: The trajectory followed by the Eef
        :param target_pose SE3: The target pose
        :param obstacles Obstacles: The obstacles in the scene
        :param target_volume Union[Cuboid, Cylinder, Sphere]: The target volume for
                                                              the trajectory
        :param target_negative_volumes Obstacles: Volumes that the target should
                                                  definitely be outside
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
            position_error=self.check_final_position(eef_trajectory[-1], target_pose),
            orientation_error=self.check_final_orientation(
                eef_trajectory[-1].so3, target_pose.so3
            ),
            eef_position_path_length=eef_pos_path,
            eef_orientation_path_length=eef_orien_path,
            num_steps=len(trajectory),
        )
        metrics.success = (
            metrics.position_error < 1
            and self.check_final_region(
                eef_trajectory[-1],
                target_volume,
                correct_negative_volumes(target_negative_volumes, target_pose),
            )
            and metrics.orientation_error < 15
            and not metrics.physical_violation
        )

    @staticmethod
    def metrics(group: List[TrajectoryMetrics]) -> TrajectoryGroupMetrics:
        """
        Calculates the metrics for a specific group

        :param group Dict[str, Any]: The group of results
        :rtype Dict[str, float]: The metrics
        """
        unskipped = [m for m in group if not m.skip]
        successes = [m for m in unskipped if m.success]
        return TrajectoryGroupMetrics(
            group_size=len(group),
            success=percent_true([m.success for m in unskipped]),
            skips=len([m for m in group if m.skip]),
            time=make_stat([m.time for m in successes]),
            step_time=make_stat([m.time / m.num_steps for m in successes]),
            env_collision_rate=percent_true([m.collision for m in unskipped]),
            self_collision_rate=percent_true([m.self_collision for m in unskipped]),
            joint_violation_rate=percent_true(
                [m.joint_limit_violation for m in unskipped]
            ),
            physical_violation_rate=percent_true(
                [m.physical_violation for m in unskipped]
            ),
            within_one_cm_rate=percent_true([m.position_error < 1 for m in unskipped]),
            within_five_cm_rate=percent_true([m.position_error < 5 for m in unskipped]),
            within_fifteen_deg_rate=percent_true(
                [m.orientation_error < 15 for m in unskipped]
            ),
            within_thirty_deg_rate=percent_true(
                [m.orientation_error < 30 for m in unskipped]
            ),
            eef_position_path_length=make_stat(
                [m.eef_position_path_length for m in successes]
            ),
            eef_orientation_path_length=make_stat(
                [m.eef_orientation_path_length for m in successes]
            ),
        )

    @staticmethod
    def print_metrics(group: List[TrajectoryMetrics]):
        """
        Prints the metrics in an easy to read format

        :param group Dict[str, float]: The group of results
        """
        metrics = Evaluator.metrics(group)
        print(f"Total problems: {metrics.group_size}")
        print(f"# Skips (Hard Failures): {metrics.skips}")
        print(f"% Success: {metrics.success:4.2f}")
        print(f"% Within 1cm: {metrics.within_one_cm_rate:4.2f}")
        print(f"% Within 5cm: {metrics.within_five_cm_rate:4.2f}")
        print(f"% Within 15deg: {metrics.within_fifteen_deg_rate:4.2f}")
        print(f"% Within 30deg: {metrics.within_thirty_deg_rate:4.2f}")
        print(f"% With Environment Collision: {metrics.env_collision_rate:4.2f}")
        print(f"% With Self Collision: {metrics.self_collision_rate:4.2f}")
        print(f"% With Joint Limit Violations: {metrics.joint_violation_rate:4.2f}")
        print(f"% With Self Collision: {metrics.self_collision_rate:4.2f}")
        print(f"% With Physical Violations: {metrics.physical_violation_rate:4.2f}")
        print(
            "Average End Eef Position Path Length:"
            f" {metrics.eef_position_path_length.mean:4.2f}"
            f" ± {metrics.eef_position_path_length.std:4.2f}"
        )
        print(
            "Average End Eef Orientation Path Length:"
            f" {metrics.eef_orientation_path_length.mean:4.2f}"
            f" ± {metrics.eef_orientation_path_length.std:4.2f}"
        )
        print(f"Average Time: {metrics.time.mean:4.2f} ± {metrics.time.std:4.2f}")
        print(
            "Average Time Per Step (Not Always Valuable):"
            f" {metrics.step_time.mean:4.6f}"
            f" ± {metrics.step_time.std:4.6f}"
        )

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
