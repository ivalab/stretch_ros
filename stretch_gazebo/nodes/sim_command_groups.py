#! /usr/bin/env python3
import itertools

import rospy
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from tf.transformations import euler_from_quaternion

import numpy as np

import hello_helpers.hello_misc as hm

KEY_ORDER = ['position', 'velocity', 'acceleration', 'contact_threshold']

def unpack_state(msg: JointTrajectoryControllerState):
    state = msg.actual
    data = itertools.zip_longest(msg.joint_names,
        state.positions, state.velocities, state.accelerations, state.effort,
        fillvalue=0
    )
    return {
        dat[0]: dict(zip(KEY_ORDER, dat[1:]))
        for dat in data 
    }


class SimCommandGroup:
    """Emulate a hardware strech robot's CommandGroup object. kinda.

    The API is slightly different (and actually more general) because I think the way hello robot wrote their API is insane.
    These commandgroups support multiple joints by default, and pub/sub to appropriate ROS topics for control and feedback.
    """

    def __init__(self, topic_root, joint_names):
        read_topic = topic_root + "/state"
        write_topic = topic_root + "/command"
        self.joint_state = None
        self.joint_names = joint_names
        self.state_sub = rospy.Subscriber(read_topic, JointTrajectoryControllerState, self._state_sub)
        self.command_pub = rospy.Publisher(write_topic, JointTrajectory)
        self.goals = {n: {"position": None, "velocity": None, "acceleration": None, "contact_threshold": None} for n in self.joint_names}

        # TODO: respect actual joint limits
        self.ranges = {n: (-float('inf'), float('inf')) for n in self.joint_names}
        # TODO: and joint limits
        self.acceptable_joint_errors = {n: 0.015 for n in self.joint_names}
        self.pub_msg = None
        self.errors = None

    def _state_sub(self, msg):
        self.joint_state = unpack_state(msg)

    def check_should_move(self, commanded_joint_names, invalid_joints_callback):
        """
        Return:     True/False for should move, or None for error
        """
        for joint in commanded_joint_names:
            if joint in self.joint_names:
                return True
        return False

    def set_goal(self, joint_goals, invalid_goal_callback, fail_out_of_range_goal, **kwargs):
        """Sets goal for the joint group
        Mostly copied from `stretch_ros/hello_helpers/src/hello_helpers/simple_command_group.py`.

        Sets and validates the goal point for the joints
        in this command group.

        Parameters
        ----------
        joint_goals: nested dict
            with keys: joint names
            with entries: dict
                with keys: 'position', 'velocity',
                           'acceleration', 'contact_threshold'
                with entries: the values (floats)
        invalid_goal_callback: func
            error callback for invalid goal
        fail_out_of_range_goal: bool
            whether to bound out-of-range goals to range or fail

        Returns
        -------
        bool
            False if commanded goal invalid, else True
        """

        goals = {}
        joints = []
        for joint_name, goal in joint_goals.items():
            if joint_name not in self.joint_names:
                continue
            goal['position'] = hm.bound_ros_command(self.ranges[joint_name], goal['position'], fail_out_of_range_goal)
            if goal['position'] is None:
                err_str = ("Received {0} goal point that is out of bounds. "
                            "Range = {1}, but goal point = {2}.").format(joint_name, self.ranges[joint_name], goal['position'])
                invalid_goal_callback(err_str)
                return False
            goals[joint_name] = goal
            joints.append(joint_name)

        positions, velocities, accelerations, efforts = list(
            zip(*[[goals[n][k] for k in KEY_ORDER] for n in joints])
        )

        msg = JointTrajectory()
        msg.joint_names = joints
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = velocities
        point.accelerations = accelerations
        point.effort = efforts
        point.time_from_start = rospy.Duration(kwargs.get('time', 1.0))
        msg.points = [point]
        self.pub_msg = msg

        self.goals = goals

        return True

    def flush_goal(self):
        self.command_pub.publish(self.pub_msg)

    def update_execution(self, **kwargs):
        state = self.joint_state
        self.errors = []
        for joint, goal in self.goals.items():
            err = abs(state[joint]['position'] - goal['position'])
            self.errors.append((joint, err))
        return self.errors

    def goal_reached(self):
        for joint, err in self.errors:
            if err > self.acceptable_joint_errors[joint]:
                return False
        return True


SIM_ARM_JOINTS = [
    "joint_arm_l3",
    "joint_arm_l2",
    "joint_arm_l1",
    "joint_arm_l0"
]
class SimArmCommandGroup(SimCommandGroup):

    def __init__(self, topic_root):
        super().__init__(topic_root, [
            "joint_lift",
            "joint_arm",
            "joint_wrist_yaw"
        ] + SIM_ARM_JOINTS )

    def _state_sub(self, msg):
        super()._state_sub(msg)
        self.joint_state['joint_arm'] = {k: sum(self.joint_state[j][k] for j in SIM_ARM_JOINTS) for k in KEY_ORDER}
        for j in SIM_ARM_JOINTS:
            del self.joint_state[j]

    def check_should_move(self, commanded_joint_names, invalid_joints_callback):
        for j in SIM_ARM_JOINTS:
            if j in commanded_joint_names:
                err_str = f"Cannot set joint {j} in arm direction -- use 'joint_arm' or 'joint_wrist_extend'"
                invalid_joints_callback(err_str)
                return None

        if 'joint_arm' in commanded_joint_names and 'joint_wrist_extend' in commanded_joint_names:
            err_str = "Cannot set both 'joint_arm' and 'joint_wrist_extend' -- they refer to the same joint"
            invalid_joints_callback(err_str)
            return None
        return super().check_should_move(commanded_joint_names, invalid_joints_callback)


    def set_goal(self, joint_goals, invalid_goal_callback, fail_out_of_range_goal, **kwargs):
        joint_goals = joint_goals.copy()
        arm_spec = joint_goals.get('joint_arm', None) or joint_goals.get('joint_wrist_extend', None)
        if arm_spec is not None:
            for j in SIM_ARM_JOINTS:
                joint_goals[j] = {k: v/4 for k, v in arm_spec.items()}

        if not super().set_goal(joint_goals, invalid_goal_callback, fail_out_of_range_goal, **kwargs):
            return False

        for j in SIM_ARM_JOINTS:
            if j in self.goals:
                del self.goals[j]
        if arm_spec is not None:
            self.goals['joint_arm'] = arm_spec
        return True


SIM_GRIPPER_JOINTS = [
    "joint_gripper_finger_left",
    "joint_gripper_finger_right"
]
class SimGripperCommandGroup(SimCommandGroup):

    def __init__(self, topic_root):
        super().__init__(topic_root, SIM_GRIPPER_JOINTS )

    def _state_sub(self, msg):
        super()._state_sub(msg)
        self.joint_state['gripper_aperture'] = self.joint_state['joint_gripper_finger_left']
        for j in SIM_GRIPPER_JOINTS:
            del self.joint_state[j]

    def check_should_move(self, commanded_joint_names, invalid_joints_callback):
        checks = []
        for joint_name in SIM_GRIPPER_JOINTS + ['gripper_aperture']:
            if joint_name in commanded_joint_names:
                checks.append(joint_name)
        if len(checks) > 1:
            err_str = f"Cannot set joints {checks} together -- they refer to the same joint"
            invalid_joints_callback(err_str)
            return None
        return super().check_should_move(commanded_joint_names, invalid_joints_callback)

    def set_goal(self, joint_goals, invalid_goal_callback, fail_out_of_range_goal, **kwargs):
        for joint_name in SIM_GRIPPER_JOINTS + ['gripper_aperture']:
            if joint_name in joint_goals:
                gripper_spec = joint_name
                break

        for j in SIM_GRIPPER_JOINTS:
            joint_goals[j] = {k: v for k, v in gripper_spec.items()}

        if not super().set_goal(joint_goals, invalid_goal_callback, fail_out_of_range_goal, **kwargs):
            return False

        for j in SIM_GRIPPER_JOINTS:
            del self.goals[j]
        self.goals['gripper_aperture'] = gripper_spec
        return True

class SimBaseCommandGroup(SimCommandGroup):
    def __init__(self, topic_root):
        read_topic = topic_root + "/odom"
        write_topic = topic_root + "/cmd_vel"
        self.joint_names = ['translate_mobile_base', 'rotate_mobile_base']
        self.joint_state = None # x, y, w
        self.state_sub = rospy.Subscriber(read_topic, Odometry, self._state_sub)
        self.command_pub = rospy.Publisher(write_topic, Twist)
        self.cmd = None
        self.goal = None
        self.start_pose = None  # x, y, w
        self.k_linear = 1
        self.k_angular = 1
        self.acceptable_joint_errors = { 'translate_mobile_base': 0.05, 'rotate_mobile_base': 0.015 }
        self.odom_raw = None

    def check_should_move(self, commanded_joint_names, invalid_joints_callback):
        checks = []
        for joint_name in SIM_GRIPPER_JOINTS + ['gripper_aperture']:
            if joint_name in commanded_joint_names:
                checks.append(joint_name)
        if len(checks) > 1:
            err_str = "SimBaseCommandGroup: Both translate_mobile_base and rotate_mobile_base commands received. Only one can be given at a time. They are mutually exclusive."
            invalid_joints_callback(err_str)
            return None
        return super().check_should_move(commanded_joint_names, invalid_joints_callback)

    def set_goal(self, joint_goals, invalid_goal_callback, fail_out_of_range_goal, **kwargs):
        for joint_name in joint_names:
            if joint_name in joint_goals:
                self.cmd = joint_name
                break

        goal = joint_goals[self.cmd]
        self.start_pose = self.joint_state
        angle = self.start_pose[2]
        if self.cmd == 'translate_mobile_base':
            vec = np.array([np.cos(angle), np.sin(angle)]) * goal['position']
            self.goal = self.start_pose[:2] + vec

        elif self.cmd == 'rotate_mobile_base':
            self.goal = angle + goal['position']

    def _state_sub(self, msg):
        self.odom_raw = msg
        pose = msg.pose.pose
        pos = pose.position
        q = pose.orientation
        angle = euler_from_quaternion((q.x, q.y, q.z, q.w))[0]
        self.joint_state = (pos.x, pos.y, angle)

    def flush_goal(self):
        pass

    def update_execution(self, **kwargs):
        state = self.joint_state
        angle = state[2]
        msg = Twist()
        if self.cmd == 'translate_mobile_base':
            vec = np.array([np.cos(angle), np.sin(angle)]) * goal['position']
            delta = self.goal - state[:2]
            delta = np.dot(vec, delta)
            cmd = self.k_linear * delta
            msg.linear.x = cmd

        elif self.cmd == 'rotate_mobile_base':
            delta = self.goal - angle
            while delta > np.pi:
                delta -= 2*np.pi
            while delta < -np.pi:
                delta += 2*np.pi
            msg.angular.z = self.k_angular * delta

        self.command_pub.publish(msg)
        self.errors = [(self.cmd, delta)]
        return self.errors
