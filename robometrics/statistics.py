from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


def percent_true(arr: Sequence) -> float:
    """
    Returns the percent true of a boolean sequence or the percent nonzero of a numerical
    sequence

    :param arr Sequence: The input sequence
    :rtype float: The percent
    """
    if len(arr) == 0:
        return 0
    return 100 * np.count_nonzero(arr) / len(arr)


@dataclass
class Statistic:
    mean: float
    std: float
    median: float
    percent_75: float
    percent_98: float

    @classmethod
    def from_list(cls, lst: Sequence[float]):
        if not lst:
            return cls(0, 0, 0, 0, 0)

        # We use np.inf as a way to filter out unwanted stats
        lst = [le for le in lst if le < np.inf]

        return Statistic(
            np.mean(lst),
            np.std(lst),
            np.median(lst),
            np.percentile(lst, 75),
            np.percentile(lst, 98),
        )

    def __str__(self):
        return (
            f"mean: {self.mean:2.3f} ± {self.std:2.3f}"
            f" median:{self.median:2.3f}"
            f" 75%: {self.percent_75:2.3f}"
            f" 98%: {self.percent_98:2.3f}"
        )


@dataclass
class TrajectoryMetrics:
    skip: bool = True
    success: bool = False
    collision: bool = True
    joint_limit_violation: bool = True
    self_collision: bool = True
    physical_violation: bool = True
    position_error: float = np.inf
    orientation_error: float = np.inf
    eef_position_path_length: float = np.inf
    eef_orientation_path_length: float = np.inf
    trajectory_length: int = 1
    attempts: int = 1
    motion_time: float = np.inf
    solve_time: float = np.inf


@dataclass
class TrajectoryGroupMetrics:
    group_size: int  # Num trajectories in group
    success: float  # Success % in group
    skips: int  # % in group with no solution
    env_collision_rate: float  # % in group with environment collisions
    self_collision_rate: float  # % in group with self collisions
    joint_violation_rate: float  # % in group with a joint limit violation
    physical_violation_rate: float  # % in group with any physical violation
    within_one_cm_rate: float  # % in group that gets within 1cm of target
    within_five_cm_rate: float  # % in group that gets within 5cm of target
    within_fifteen_deg_rate: float  # % in group that gets within 15deg of target
    within_thirty_deg_rate: float  # % in group that gets within 30deg of target
    eef_position_path_length: Statistic  # Stats on Euclidean eef path length
    eef_orientation_path_length: Statistic  # Stats on orientation path length (deg)
    attempts: Statistic  # Stats on the number of attempts taken to reach a solution
    position_error: Statistic  # Stats on position error in reaching target
    orientation_error: Statistic  # Stats on orientation error
    motion_time: Statistic  # Stats on time to execute (useful for evaluating time-optimality of trajectories)
    solve_time: Statistic  # Stats on time to find a trajectory
    solve_time_per_step: Statistic  # Stats on average time to find an action (meaningful to evaluate reaction time)

    @classmethod
    def from_list(cls, group: List[TrajectoryMetrics]):
        unskipped = [m for m in group if not m.skip]
        successes = [m for m in unskipped if m.success]
        return cls(
            group_size=len(group),
            success=percent_true([m.success for m in group]),
            skips=len([m for m in group if m.skip]),
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
            eef_position_path_length=Statistic.from_list(
                [m.eef_position_path_length for m in successes]
            ),
            eef_orientation_path_length=Statistic.from_list(
                [m.eef_orientation_path_length for m in successes]
            ),
            attempts=Statistic.from_list([m.attempts for m in successes]),
            position_error=Statistic.from_list([m.position_error for m in successes]),
            orientation_error=Statistic.from_list(
                [m.orientation_error for m in successes]
            ),
            motion_time=Statistic.from_list([m.motion_time for m in successes]),
            solve_time=Statistic.from_list([m.solve_time for m in successes]),
            solve_time_per_step=Statistic.from_list(
                [m.solve_time / m.trajectory_length for m in successes]
            ),
        )

    def print_summary(self):
        print(f"Total problems: {self.group_size}")
        print(f"# Skips (Hard Failures): {self.skips}")
        print(f"% Success: {self.success:4.2f}")
        print(f"% Within 1cm: {self.within_one_cm_rate:4.2f}")
        print(f"% Within 5cm: {self.within_five_cm_rate:4.2f}")
        print(f"% Within 15deg: {self.within_fifteen_deg_rate:4.2f}")
        print(f"% Within 30deg: {self.within_thirty_deg_rate:4.2f}")
        print(f"% With Environment Collision: {self.env_collision_rate:4.2f}")
        print(f"% With Self Collision: {self.self_collision_rate:4.2f}")
        print(f"% With Joint Limit Violations: {self.joint_violation_rate:4.2f}")
        print(f"% With Self Collision: {self.self_collision_rate:4.2f}")
        print(f"% With Physical Violations: {self.physical_violation_rate:4.2f}")
        print(
            "Average End Eef Position Path Length:"
            f" {self.eef_position_path_length.mean:4.2f}"
            f" ± {self.eef_position_path_length.std:4.2f}"
        )
        print(
            "Average End Eef Orientation Path Length:"
            f" {self.eef_orientation_path_length.mean:4.2f}"
            f" ± {self.eef_orientation_path_length.std:4.2f}"
        )
        print(
            f"Average Motion Time: {self.motion_time.mean:4.2f} ± {self.motion_time.std:4.2f}"
        )
        print(
            f"Average Solve Time: {self.solve_time.mean:4.2f} ± {self.solve_time.std:4.2f}"
        )
        print(
            "Average Time Per Step (Not Always Valuable):"
            f" {self.solve_time_per_step.mean:4.6f}"
            f" ± {self.solve_time_per_step.std:4.6f}"
        )