from typing import Tuple

import numpy as np
from geometrout import SE3, SO3

from robometrics.robometrics_types import EefTrajectory, Obstacles, Trajectory
from robometrics.robot import Robot


def position_error_in_cm(final_pose: SE3, target: SE3) -> float:
    """
    Gets the number of centimeters between the final pose and target

    :param final_pose SE3: The final pose of the trajectory
    :param target SE3: The target pose
    :rtype float: The distance in centimeters
    """
    return 100 * np.linalg.norm(final_pose.pos - target.pos)


def orientation_error_in_degrees(final_orientation: SO3, target: SO3) -> float:
    """
    Gets the number of degrees between the final orientation and the target
    orientation

    :param final_orientation SO3: The final orientation
    :param target SO3: The final target orientation
    :rtype float: The rotational distance in degrees
    """
    return np.abs((final_orientation * target.conjugate).degrees)


def has_self_collision(
    robot: Robot,
    trajectory: Trajectory,
) -> bool:
    """
    Checks whether there is a self collision (using OR between different methods)

    :param trajectory Trajectory: The trajectory
    :rtype bool: Whether there is a self collision
    """
    for q in trajectory:
        if robot.has_self_collision(q):
            return True
    return False


def in_collision(
    robot: Robot,
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
        if robot.has_scene_collision(q, obstacles):
            return True
    return False


def violates_joint_limits(robot: Robot, trajectory: Trajectory) -> bool:
    """
    Checks whether any configuration in the trajectory violates joint limits

    :param trajectory Trajectory: The trajectory
    :rtype bool: Whether there is a joint limit violation
    """
    for q in trajectory:
        if not robot.within_joint_limits(q):
            return True
    return False


def has_physical_violation(
    robot: Robot, trajectory: Trajectory, obstacles: Obstacles
) -> bool:
    """
    Checks whether there is any physical violation
    Checks collision, self collision, joint limit violation

    :param trajectory Trajectory: The trajectory
    :param obstacles Obstacles: The obstacles in the scene
    :rtype bool: Whether there is at least one physical violation
    """
    return (
        in_collision(robot, trajectory, obstacles)
        or violates_joint_limits(robot, trajectory)
        or has_self_collision(robot, trajectory)
    )


def calculate_eef_path_lengths(eef_trajectory: EefTrajectory) -> Tuple[float, float]:
    """
    Calculate the end effector path lengths (position and orientation).
    Orientation is in degrees.

    :param trajectory Trajectory: The trajectory
    :rtype Tuple[float, float]: The path lengths (position, orientation)
    """

    eef_positions = np.asarray([pose.pos for pose in eef_trajectory])
    assert eef_positions.ndim == 2 and eef_positions.shape[1] == 3
    position_step_lengths = np.linalg.norm(np.diff(eef_positions, 1, axis=0), axis=1)
    eef_position_path_length = sum(position_step_lengths)

    eef_so3 = [pose.so3 for pose in eef_trajectory]
    eef_orientation_path_length = 0
    for qi, qj in zip(eef_so3[:-1], eef_so3[1:]):
        eef_orientation_path_length += np.abs((qj * qi.conjugate).degrees)
    return eef_position_path_length, eef_orientation_path_length
