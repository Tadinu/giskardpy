import gc
import time
import traceback
from collections import defaultdict
from datetime import datetime
from itertools import combinations, chain
from typing import Dict, List, Callable, Tuple, Optional, TYPE_CHECKING

import giskardpy.casadi_wrapper as cas
import numpy as np
import pandas as pd
import pytest
import urdf_parser_py.urdf as up
from giskardpy.data_types.data_types import PrefixName, Derivatives, ColorRGBA
from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.god_map import GodMap
from giskardpy.model.better_pybullet_syncer import BetterPyBulletSyncer
from giskardpy.model.collision_avoidance_config import DefaultCollisionAvoidanceConfig
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.joints import OmniDrive, PrismaticJoint
from giskardpy.model.links import Link, BoxGeometry
from giskardpy.model.trajectory import Trajectory
from giskardpy.model.utils import hacky_urdf_parser_fix
from giskardpy.model.world import WorldTree
from giskardpy.model.world_config import EmptyWorld, WorldWithFixedRobot, WorldWithOmniDriveRobot
from giskardpy.motion_statechart.monitors.cartesian_monitors import PoseReached
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.constraint import EqualityConstraint, InequalityConstraint, DerivativeInequalityConstraint
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller import QPFormulation
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.symbol_manager import symbol_manager
from giskardpy.utils.utils import suppress_stderr
from giskardpy.model.collision_avoidance_config import CollisionAvoidanceConfig
from giskardpy.model.collision_world_syncer import CollisionCheckerLib
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointVelocity
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE
from giskardpy.qp.constraint import DerivativeEqualityConstraint
from giskardpy.utils.math import limit
from test.utils_for_tests import pr2_urdf
from giskardpy.user_interface import GiskardWrapper

import mujoco

class MuJoCoSim(GiskardWrapper):
    def __init__(self, model: mujoco.MjModel,
                 data: mujoco.MjData,
                 world: WorldTree, control_dt: float, mpc_dt: float, h: int, solver: SupportedQPSolver,
                 jerk_limit: float,
                 alpha: float, graph_styles: Optional[List[Tuple[str, str]]] = None,
                 qp_formulation: QPFormulation = QPFormulation.implicit):
        super().__init__(world=world, control_dt=control_dt, mpc_dt=mpc_dt,h=h,solver=solver,
                         jerk_limit=jerk_limit, alpha=alpha, graph_styles=graph_styles,qp_formulation=qp_formulation)
        self.mj_model = model
        self.mj_data = data

    def update(self, q: Optional[np.ndarray] = None, kinematics_only: bool = True) -> None:
        """Run forward kinematics.

        Args:
            q: Optional configuration vector to override internal `data.qpos` with.
            kinematics_only: If True, only compute kinematic quantities. Else, a full mj_step is done
        """
        if q is not None:
            self.mj_data.qpos = q
        # The minimal function call required to get updated frame transforms is
        # mj_kinematics. An extra call to mj_comPos is required for updated Jacobians.
        if kinematics_only:
            mujoco.mj_kinematics(self.mj_model, self.mj_data)
            mujoco.mj_comPos(self.mj_model, self.mj_data)
        else:
            mujoco.mj_step(self.mj_model, self.mj_data)