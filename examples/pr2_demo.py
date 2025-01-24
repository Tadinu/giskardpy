from pathlib import Path
import os

from typing_extensions import Optional, List, Tuple
import numpy as np
import traceback

from mujoco import MjModel, mj_saveModel, mj_saveLastXML
import mujoco.viewer
from dm_control import mjcf
from loop_rate_limiters import RateLimiter

from giskardpy.data_types.data_types import PrefixName, Derivatives, ColorRGBA
from giskardpy.model.world import WorldTree
from giskardpy.god_map import god_map
from giskardpy.mujoco_sim import MuJoCoSim
from giskardpy.qp.qp_controller import QPFormulation
from giskardpy.qp.qp_solver_ids import SupportedQPSolver
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPoseAsTask, CartesianPosition, CartesianPositionVelocityGoal
from giskardpy.motion_statechart.tasks.joint_tasks import JointVelocity
from giskardpy.motion_statechart.tasks.task import WEIGHT_BELOW_CA, WEIGHT_COLLISION_AVOIDANCE

from test.test_giskard_library import PR2CollisionAvoidance, GoalSequence

_HERE = Path(__file__).parent
_XML = _HERE / "pr2" / "pr2.xml"
mj_model = mujoco.MjModel.from_xml_path(_XML.as_posix())
mj_data = mujoco.MjData(mj_model)

from giskardpy.model.world_config import EmptyWorld, WorldWithFixedRobot, WorldWithOmniDriveRobot
from test.utils_for_tests import pr2_urdf

sim_time = 6
noise = 0.00
control_dt = 0.01
pos_noise = 0.001
vel_noise = 0
acc_noise = 0
graph_styles = [
    [':', 'black'],
    ['none', 'black'],
    ['none', 'black'],
    ['!shade above', 'red'],
    ['!shade above', 'red'],
    ['!shade below', 'red'],
    ['!shade above', 'red'],
    ['!shade below', 'red'],
    ['-', '#003399'],
    ['-', '#003399'],
    ['-', '#003399'],
    ['-', '#003399'],
]
goals = [
    [0, -0.1],
    # [0.8, -0.15]
]
x = np.linspace(0.8, sim_time, 1000)
x2 = np.linspace(0.75, 1.75, 1000)
y = np.cos((x) * np.pi * x2 * 0.8) * 0.1 - 0.2
goals.extend(list(zip(x.tolist(), y.tolist())))
goal_f = GoalSequence(goals, noise)

def pr2_world() -> WorldTree:
    urdf = open('urdfs/pr2.urdf', 'r').read()
    config = WorldWithOmniDriveRobot(urdf=urdf)
    with config.world.modify_world():
        config.setup()
    config.world.register_controlled_joints(config.world.movable_joint_names)
    collision_avoidance = PR2CollisionAvoidance()
    collision_avoidance.setup()
    return config.world

# test_joint_goal_pr2_pos_limits
def create_sim() -> MuJoCoSim:
    sim = MuJoCoSim(model=mj_model, data=mj_data, world=pr2_world(),
                    control_dt=control_dt,
                    mpc_dt=control_dt,
                    h=7,
                    solver=SupportedQPSolver.qpalm,
                    alpha=0.1,
                    graph_styles=graph_styles,
                    jerk_limit=None,
                    qp_formulation=QPFormulation.explicit_no_acc)
    sim.reset()
    first_goal, first_weight = goal_f('', 0)
    sim.add_joint_goal(joint_names=('pr2/r_elbow_flex_joint',), name='g1',
                       weight=first_weight, goal=first_goal)
    sim.compile()
    return sim

# test_joint_vel_goal_pr2
def create_sim2() -> MuJoCoSim:
    jerk_limit = None
    h = 7
    control_dt = 0.01
    goal1 = 0.14
    goal2 = 0.5
    goal3 = 0.7
    graph_styles = [
        ['--', 'black'],
        [':', 'black'],
        [':', 'black'],
        ['-', '#003399'],
        ['-', '#003399'],
        ['-', '#003399'],
        ['-', '#003399'],

        ['--', '#993000'],
        [':', '#993000'],
        [':', '#993000'],

        ['--', '#993000'],
        [':', '#993000'],
        [':', '#993000'],
    ]
    sim = MuJoCoSim(model=mj_model, data=mj_data, world=pr2_world(),
                    control_dt=control_dt,
                    mpc_dt=control_dt,
                    h=h,
                    solver=SupportedQPSolver.qpalm,
                    alpha=0.1,
                    graph_styles=graph_styles,
                    jerk_limit=jerk_limit,
                    qp_formulation=QPFormulation.explicit_no_acc)
    sim.reset()
    sim.add_joint_goal(
        joint_names=('pr2/r_wrist_roll_joint',),
        name='t1',
        weight=1,
        goal=goal1)
    sim.add_joint_goal(
        joint_names=('pr2/r_wrist_roll_joint',),
        name='t2',
        weight=1,
        goal=goal2)
    sim.add_joint_goal(
        joint_names=('pr2/r_wrist_roll_joint',),
        name='g3',
        weight=1,
        goal=goal3)
    sim.compile()

    def goal_function(name, time):
        if name == 't1':
            return goal1, WEIGHT_BELOW_CA
        if name == 't2':
            g2_weight = WEIGHT_COLLISION_AVOIDANCE
            if time > 1:
                g2_weight = WEIGHT_BELOW_CA
            if time > 2:
                g2_weight = 0
            if time > 3:
                g2_weight = WEIGHT_BELOW_CA
            if time > 4:
                g2_weight = WEIGHT_COLLISION_AVOIDANCE
            if time > 4.2:
                g2_weight = WEIGHT_BELOW_CA
            if time > 5:
                g2_weight = WEIGHT_BELOW_CA
            return goal2, g2_weight
        if name == 'g3':
            return goal3, WEIGHT_BELOW_CA

    global goal_f
    goal_f = goal_function
    return sim

def create_sim3() -> MuJoCoSim:
    pr2 = pr2_world()
    goal = 0.0
    pr2.update_default_weights({Derivatives.velocity: 0.01,
                                Derivatives.acceleration: 0.0,
                                Derivatives.jerk: 0.0})
    pr2.update_default_limits({
        Derivatives.velocity: 1,
        Derivatives.acceleration: np.inf,
        Derivatives.jerk: 100
    })
    sim = MuJoCoSim(model=mj_model, data=mj_data, world=pr2,
                    control_dt=0.05,
                    mpc_dt=0.05,
                    h=9,
                    jerk_limit=2500,
                    solver=SupportedQPSolver.qpalm,
                    alpha=.1,
                    qp_formulation=QPFormulation.implicit)
    joint_names = pr2.movable_joint_names[:1]
    # joint_names=pr2.movable_joint_names[14:15],
    # joint_names=pr2.movable_joint_names[:35],
    # joint_names=pr2.movable_joint_names[27:31],
    # joint_names=pr2.movable_joint_names[:-1],
    # joint_names=[
    # pr2.movable_joint_names[27],
    # pr2.movable_joint_names[13]
    # ],
    sim.add_joint_vel_goal(goal=goal, joint_names=joint_names)
    sim.compile()
    return sim


if __name__ == "__main__":
    sim = create_sim3()

    plot = False
    plot_legend=True
    plot_kwargs={'unit': 'rad', 'file_name': 'pos_limits.pdf'}
    with mujoco.viewer.launch_passive(model=mj_model, data=mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)
        #mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        rate = RateLimiter(frequency=100.0, warn=False)

        np.random.seed(69)
        failed = False
        pos_control = False
        qp_times = []
        try:
            while viewer.is_running():
                total_time, parameter_time, qp_time, update_world_time, collision_time = sim.step()
                sim.apply_noise(pos_noise, vel_noise, acc_noise)
                qp_times.append(qp_time)

                if pos_control:
                    for goal_name in sim.goal_state:
                        next_goal, next_weight = goal_f(goal_name, god_map.time)
                        sim.update_goal(goal_name, next_goal, next_weight)
                else:
                    goal = np.sin(god_map.time * 8) * 0.5
                    sim.update_goal("goal", goal, 1)

                for i in range(mj_model.nq):
                    joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    #actuator = mj_model.actuator(joint_name)
                    mj_data.qpos[i] = sim.world.state[joint_name].position
                sim.update(kinematics_only=pos_control)
                mujoco.mj_camlight(mj_model, mj_data)

                # Visualize at fixed FPS.
                viewer.sync()
                rate.sleep()
        except Exception as e:
            traceback.print_exc()
            print(e)
            failed = True
            raise e
        finally:
            if plot:
                sim.plot_traj(plot_kwargs, plot_legend)
                avg = np.average(qp_times)
                print(f'avg time {avg} or {1 / avg}hz')
                traj_dict = sim.traj.to_dict(normalize_position=False, filter_0_vel=False, sort=True)