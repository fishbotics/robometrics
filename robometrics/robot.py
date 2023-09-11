from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import urchin
import yaml
from geometrout.utils import transform_in_place


@dataclass
class CollisionSphereConfig:
    collision_spheres: Dict[str, Dict[str, Any]]
    self_collision_ignore: Dict[str, List[str]]
    self_collision_buffer: Dict[str, float]

    @staticmethod
    def load_from_file(file_path):
        with open(file_path) as f:
            collision_sphere_config = yaml.safe_load(f)
        return CollisionSphereConfig(
            collision_sphere_config["collision_spheres"],
            collision_sphere_config["self_collision_ignore"],
            collision_sphere_config["self_collision_buffer"],
        )


class Robot:
    def __init__(
        self,
        urdf_path,
        collision_sphere_config: CollisionSphereConfig,
    ):
        self.urdf = urchin.URDF.load(urdf_path, lazy_load_meshes=True)
        self.actuated_joints = [
            j for j in self.urdf.joints if j.joint_type != "fixed" and j.mimic is None
        ]
        self.collision_sphere_config = collision_sphere_config

        self._init_collision_spheres()
        self._init_self_collision_spheres()

    def _init_collision_spheres(self):
        link_names = set([lnk.name for lnk in self.urdf.links])
        for link_name in self.collision_sphere_config.collision_spheres:
            if link_name not in link_names:
                logging.warning(
                    "Collision sphere description file has"
                    f" link ({link_name}) not found in URDF"
                )

    def _init_self_collision_spheres(self):
        link_names = set([lnk.name for lnk in self.urdf.links])
        radii = []
        cspheres = self.collision_sphere_config.collision_spheres
        buffers = self.collision_sphere_config.self_collision_buffer
        ignores = self.collision_sphere_config.self_collision_ignore

        for link_name, props in cspheres.items():
            if link_name not in link_names:
                logging.warning(
                    "Self Collision sphere description file"
                    f" has link ({link_name}) not found in URDF"
                )
                continue
            radii.extend([(link_name, p["radius"] + buffers[link_name]) for p in props])

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

    def within_joint_limits(self, q):
        # Find all the joints that are actively controlled, according to the URDF
        self.ensure_dof(q)
        for qi, joint in zip(q, self.actuated_joints):
            if qi < joint.lower or qi > joint.upper:
                return False
        return True

    def _fk_sphere_info(self, q):
        self.ensure_dof(q)
        fk = self.urdf.link_fk(q, use_names=True)
        centers, radii = [], []
        spheres = self.collision_sphere_config.collision_spheres
        for link_name, props in spheres.items():
            if link_name not in fk:
                continue
            centers.append(
                transform_in_place(
                    np.stack([s["center"] for s in props]), fk[link_name]
                )
            )
            radii.append(np.stack([s["radius"] for s in props]))
        return np.concatenate(centers, axis=0), np.concatenate(radii, axis=0)

    def has_self_collision(self, q):
        centers, _ = self._fk_sphere_info(q)
        centers_matrix = np.tile(centers, (centers.shape[0], 1, 1))
        pairwise_distances = np.linalg.norm(
            centers_matrix - centers_matrix.transpose((1, 0, 2)), axis=2
        )
        return np.any(pairwise_distances < self.self_collision_distance_matrix)

    def has_scene_collision(self, q, obstacles: List[Any]):
        centers, radii = self._fk_sphere_info(q)
        for o in obstacles:
            if np.any(o.sdf(centers) < radii):
                return True
        return False
