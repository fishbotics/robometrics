from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import urchin
import yaml
from geometrout import Sphere
from geometrout.maths import transform_in_place

from robometrics.robometrics_types import Obstacles


@dataclass
class CollisionSphereConfig:
    """Used to represent the collision spheres for the robot."""

    collision_spheres: Dict[str, List[Sphere]]
    self_collision_ignore: Dict[str, List[str]]
    self_collision_buffer: Dict[str, float]

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path) as f:
            collision_sphere_config = yaml.safe_load(f)
        return cls(
            collision_spheres={
                k: [Sphere(center=np.ravel(s["center"]), radius=s["radius"]) for s in v]
                for k, v in collision_sphere_config["collision_spheres"].items()
            },
            self_collision_ignore=collision_sphere_config["self_collision_ignore"],
            self_collision_buffer=collision_sphere_config["self_collision_buffer"],
        )

    @classmethod
    def load_from_dictionary(cls, data_dictionary):
        return cls(
            collision_spheres={
                k: [Sphere(center=np.ravel(s["center"]), radius=s["radius"]) for s in v]
                for k, v in data_dictionary["collision_spheres"].items()
            },
            self_collision_ignore=data_dictionary["self_collision_ignore"],
            self_collision_buffer=data_dictionary["self_collision_buffer"],
        )


class Robot:
    """Used to represent a robot (defined by a URDF) and its collision spheres."""

    def __init__(
        self,
        urdf_path: Union[Path, str],
        collision_sphere_config: CollisionSphereConfig,
    ):
        """Loads the URDF and collision spheres.

        Args:
            urdf_path: Path to the URDF file.
            collision_sphere_config: Path to collision sphere yaml file.
        """
        self.urdf = urchin.URDF.load(urdf_path, lazy_load_meshes=True)
        self.actuated_joints = [
            j for j in self.urdf.joints if j.joint_type != "fixed" and j.mimic is None
        ]
        self.collision_sphere_config = collision_sphere_config

        self._check_collision_spheres()
        self._init_self_collision_spheres()

    def _check_collision_spheres(self):
        """Verifies that the links in the collision sphere file match the URDF."""
        link_names = set([lnk.name for lnk in self.urdf.links])
        for link_name in self.collision_sphere_config.collision_spheres:
            if link_name not in link_names:
                logging.warning(
                    "Collision sphere description file has"
                    f" link ({link_name}) not found in URDF"
                )

    def _init_self_collision_spheres(self):
        """Creates a matrix for self-collision checking."""
        link_names = set([lnk.name for lnk in self.urdf.links])
        radii = []
        cspheres = self.collision_sphere_config.collision_spheres
        buffers = self.collision_sphere_config.self_collision_buffer
        ignores = self.collision_sphere_config.self_collision_ignore

        for link_name, link_spheres in cspheres.items():
            if link_name not in link_names:
                logging.warning(
                    "Self Collision sphere description file"
                    f" has link ({link_name}) not found in URDF"
                )
                continue
            radii.extend(
                [(link_name, s.radius + buffers[link_name]) for s in link_spheres]
            )

        nspheres = len(radii)
        mat = -np.inf * np.ones((nspheres, nspheres))
        for ii, (ln1, rad1) in enumerate(radii):
            for jj, (ln2, rad2) in enumerate(radii):
                if ln1 == ln2:
                    continue
                if ln2 in ignores.get(ln1, []) or ln1 in ignores.get(ln2, []):
                    continue
                mat[ii, jj] = rad1 + rad2
        self.self_collision_distance_matrix = mat

    @property
    def dof(self):
        return len(self.actuated_joints)

    def ensure_dof(self, q):
        assert (
            len(q) == self.dof
        ), f"q (length: {len(q)}) must have length equal to robot DOF ({self.dof})"

    def within_joint_limits(self, q: np.ndarray) -> bool:
        # Find all the joints that are actively controlled, according to the URDF
        """Checks whether robot is within joint limits.

        Args:
            q: The robot configuration

        Returns:
            Whether the robot configuration is within the published joint limits
        """
        self.ensure_dof(q)
        for qi, joint in zip(q, self.actuated_joints):
            if qi < joint.limit.lower or qi > joint.limit.upper:
                return False
        return True

    def _fk_sphere_info(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Maps the spheres to the robot with forward kinematics.

        Args:
            q: The robot configuration

        Returns:
            A tuple of numpy arrays: (centers, radii)
        """
        self.ensure_dof(q)
        fk = self.urdf.link_fk(q, use_names=True)
        centers, radii = [], []
        spheres = self.collision_sphere_config.collision_spheres
        for link_name, link_spheres in spheres.items():
            if link_name not in fk:
                continue
            centers.append(
                transform_in_place(
                    np.stack([s.center for s in link_spheres]), fk[link_name]
                )
            )
            radii.append(np.stack([s.radius for s in link_spheres]))
        return np.concatenate(centers, axis=0), np.concatenate(radii, axis=0)

    def has_self_collision(self, q: np.ndarray) -> bool:
        """Checks for illegal self-collisions.

        Uses the allowable self-collisions defined in the CollisionSphereConfig to
        check whether there are any collisions that are considered bad.

        Args:
            q: The robot configuration

        Returns:
            Whether there are illegal self collisions
        """
        centers, _ = self._fk_sphere_info(q)
        centers_matrix = np.tile(centers, (centers.shape[0], 1, 1))
        pairwise_distances = np.linalg.norm(
            centers_matrix - centers_matrix.transpose((1, 0, 2)), axis=2
        )
        return np.any(pairwise_distances < self.self_collision_distance_matrix)

    def has_scene_collision(self, q: np.ndarray, obstacles: Obstacles) -> bool:
        """Checks whether any of the robot spheres are in collision with an obstacle.

        Args:
            q: The robot configuration (must match dof)
            obstacles: The obstacles in the scene

        Returns:
            True if colliding, False if not
        """
        centers, radii = self._fk_sphere_info(q)
        for o in obstacles:
            if np.any(o.sdf(centers) < radii):
                return True
        return False
