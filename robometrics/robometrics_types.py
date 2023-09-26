import logging
from typing import List, Sequence, Union

import numpy as np

try:
    from geometrout import SE3, Cuboid, Cylinder, Sphere

    EefTrajectory = Sequence[SE3]
    Obstacles = List[Union[Cuboid, Cylinder, Sphere]]
except ImportError:
    logging.warn("Optional package geometrout not installed. Some types not available.")

Trajectory = Sequence[Union[Sequence, np.ndarray]]
