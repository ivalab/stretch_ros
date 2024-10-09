#! /usr/bin/env python3
import itertools

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryFeedback
from control_msgs.msg import FollowJointTrajectoryResult
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from std_srvs.srv import Trigger, TriggerResponse
from std_srvs.srv import SetBool, SetBoolResponse

from sim_command_groups import SimCommandGroup, KEY_ORDER, \
                               SimArmCommandGroup, \
                               SimGripperCommandGroup, \
                               SimBaseCommandGroup


class StretchGazeboAdapter:

    def __init__(self, node_name):
        self.server = actionlib.SimpleActionServer('/stretch_controller/follow_joint_trajectory',
                                                   FollowJointTrajectoryAction,
                                                   execute_cb=self.execute_cb,
                                                   auto_start=False)
        self.feedback = FollowJointTrajectoryFeedback()
        self.result = FollowJointTrajectoryResult()
        self.node_name = node_name
        self.fail_out_of_range_goal = rospy.get_param('~fail_out_of_range_goal', True)
        self.default_goal_timeout_duration = rospy.Duration(10.0)

        """
        all: state, command

        stretch_head_controller/
            io: joint_head_pan
            io: joint_head_tilt

        stretch_arm_controller/
            io: joint_lift
            io: (joint_arm, wrist_extension)
                joint_arm_l3
                joint_arm_l2
                joint_arm_l1
                joint_arm_l0
            io: joint_wrist_yaw

        stretch_gripper_controller/
            io: (joint_gripper_finger_left, joint_gripper_finger_right, gripper_aperture)
                joint_gripper_finger_left
                joint_gripper_finger_right

        special:
        stretch_diff_drive_controller/
            io: translate_mobile_base
            io: rotate_mobile_base

            loop: odom
            out: cmd_vel
        """
        self.head = SimCommandGroup("/stretch_head_controller", ["joint_head_pan", "joint_head_tilt"])
        self.arm = SimArmCommandGroup("/stretch_arm_controller")
        self.gripper = SimGripperCommandGroup("/stretch_gripper_controller")
        self.base = SimBaseCommandGroup("/stretch_diff_drive_controller")
        self.command_groups = [self.head, self.arm, self.gripper, self.base]

        self.switch_to_navigation_mode_service = rospy.Service('/switch_to_navigation_mode',
                                                               Trigger,
                                                               self.navigation_mode_service_callback)

        self.switch_to_position_mode_service = rospy.Service('/switch_to_position_mode',
                                                             Trigger,
                                                             self.position_mode_service_callback)
        
        self.stop_the_robot_service = rospy.Service('/stop_the_robot',
                                                    Trigger,
                                                    self.stop_the_robot_callback)

        self.calibrate_the_robot_service = rospy.Service('/calibrate_the_robot',
                                                         Trigger,
                                                         self.calibrate_callback)

        self.stow_the_robot_service = rospy.Service('/stow_the_robot',
                                                    Trigger,
                                                    self.stow_the_robot_callback)

        self.runstop_service = rospy.Service('/runstop',
                                              SetBool,
                                              self.runstop_service_callback)
        self.stop_the_robot = True
        self.joint_state_rate = rospy.Rate(rospy.get_param('~rate', 15.0))
        self.joint_state_pub = rospy.Publisher('/stretch/joint_states', JointState, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
        self.vel_sub = rospy.Subscriber('/stretch/cmd_vel', Twist, self.set_mobile_base_velocity_callback)
        self.server.start()


    def loop(self):
        self.joint_state_rate.sleep()

        msg = JointState()
        joint_names = []
        positions = []
        velocities = []
        efforts = []
        for part in [self.head, self.arm, self.gripper]:
            state = part.joint_state
            if state is None:
                return
            for name in part.joint_names:
                if name in state:
                    joint_names.append(name)
                    positions.append(state[name]['position'])
                    velocities.append(state[name]['velocity'])
                    # We don't do effort since some joints don't report it
        msg.name = joint_names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts

        self.joint_state_pub.publish(msg)
        if self.base.odom_raw is not None:
            self.odom_pub.publish(self.base.odom_raw)

    def set_mobile_base_velocity_callback(self, twist):
        # TODO: don't accept v command unless in navigation mode
        #if self.robot_mode != 'navigation':
        #    error_string = '{0} action server must be in navigation mode to receive a twist on cmd_vel. Current mode = {1}.'.format(self.node_name, self.robot_mode)
        #    rospy.logerr(error_string)
        #    return
        self.base.command_pub.publish(twist)

    ######## SERVICE CALLBACK EMULATION #######
    # NOTE: most of these are simply unimplemented. lol

    def stop_the_robot_callback(self, request):
        self.stop_the_robot = True

        rospy.loginfo('Received stop_the_robot service call, so commanded all actuators to stop.')
        return TriggerResponse(
            success=True,
            message='Stopped the robot.'
            )

    def calibrate_callback(self, request):
        # NOTE: not implemented
        rospy.loginfo('Received calibrate_the_robot service call.')
        return TriggerResponse(
            success=True,
            message='Calibrated.'
        )

    def stow_the_robot_callback(self, request):
        # NOTE: not implemented
        rospy.loginfo('Recevied stow_the_robot service call.')
        return TriggerResponse(
            success=True,
            message='Stowed.'
        )

    def navigation_mode_service_callback(self, request):
        # NOTE: not implemented
        return TriggerResponse(
            success=True,
            message='Now in navigation mode.'
            )

    def position_mode_service_callback(self, request):
        # NOTE: not implemented
        return TriggerResponse(
            success=True,
            message='Now in position mode.'
            )

    def runstop_service_callback(self, request):
        # NOTE: partially implemented
        if request.data:
            self.stop_the_robot = True

        return SetBoolResponse(
            success=True,
            message='is_runstopped: {0}'.format(request.data)
            )

    def execute_cb(self, goal):
        self.stop_the_robot = False

        commanded_joint_names = goal.trajectory.joint_names

        ###################################################
        # Decide what to do based on the commanded joints.
        active_parts = []
        for part in self.command_groups:
            res = part.check_should_move(commanded_joint_names, self.invalid_joints_callback)
            if res is None:
                # The joint names violated this command
                # group's requirements. The command group should have
                # reported the error.
                return
            if res:
                active_parts.append(part)

        ###################################################
        # Try to reach each of the goals in sequence until
        # an error is detected or success is achieved.
        for pointi, point in enumerate(goal.trajectory.points):
            rospy.logdebug(("{0} joint_traj action: "
                            "target point #{1} = <{2}>").format(self.node_name, pointi, point))

            data = itertools.zip_longest(commanded_joint_names,
                point.positions, point.velocities, point.accelerations, point.effort,
                fillvalue=0
            )
            goals = {
                dat[0]: dict(zip(KEY_ORDER, dat[1:]))
                for dat in data 
            }

            valid_goals = [c.set_goal(goals, self.invalid_goal_callback, self.fail_out_of_range_goal)
                           for c in active_parts]
            if not all(valid_goals):
                # At least one of the goals violated the requirements
                # of a command group. Any violations should have been
                # reported as errors by the command groups.
                return
            
            for c in active_parts:
                c.flush_goal()

            goals_reached = [False for c in active_parts]
            update_rate = rospy.Rate(15.0)
            goal_start_time = rospy.Time.now()

            while not all(goals_reached):
                if (rospy.Time.now() - goal_start_time) > self.default_goal_timeout_duration:
                    err_str = ("Time to execute the current goal point = <{0}> exceeded the "
                               "default_goal_timeout = {1}").format(point, self.default_goal_timeout_s)
                    self.goal_tolerance_violated_callback(err_str)
                    return

                # Check if a premption request has been received.
                if self.stop_the_robot or self.server.is_preempt_requested():
                    rospy.logdebug(("{0} joint_traj action: PREEMPTION REQUESTED, but not stopping "
                                    "current motions to allow smooth interpolation between "
                                    "old and new commands.").format(self.node_name))
                    self.server.set_preempted()
                    return

                named_errors = [c.update_execution()
                                for c in active_parts]

                self.feedback_callback(commanded_joint_names, point, named_errors)
                goals_reached = [c.goal_reached() for c in active_parts]
                update_rate.sleep()

            rospy.logdebug("{0} joint_traj action: Achieved target point.".format(self.node_name))

        self.success_callback("Achieved all target points.")
        return

    def invalid_joints_callback(self, err_str):
        if self.server.is_active() or self.server.is_preempt_requested():
            rospy.logerr("{0} joint_traj action: {1}".format(self.node_name, err_str))
            self.result.error_code = self.result.INVALID_JOINTS
            self.result.error_string = err_str
            self.server.set_aborted(self.result)

    def invalid_goal_callback(self, err_str):
        if self.server.is_active() or self.server.is_preempt_requested():
            rospy.logerr("{0} joint_traj action: {1}".format(self.node_name, err_str))
            self.result.error_code = self.result.INVALID_GOAL
            self.result.error_string = err_str
            self.server.set_aborted(self.result)

    def goal_tolerance_violated_callback(self, err_str):
        if self.server.is_active() or self.server.is_preempt_requested():
            rospy.logerr("{0} joint_traj action: {1}".format(self.node_name, err_str))
            self.result.error_code = self.result.GOAL_TOLERANCE_VIOLATED
            self.result.error_string = err_str
            self.server.set_aborted(self.result)

    def feedback_callback(self, commanded_joint_names, desired_point, named_errors):
        clean_named_errors = []
        for named_error in named_errors:
            if type(named_error) == tuple:
                clean_named_errors.append(named_error)
            elif type(named_error) == list:
                clean_named_errors += named_error
        clean_named_errors_dict = dict((k, v) for k, v in clean_named_errors)

        actual_point = JointTrajectoryPoint()
        error_point = JointTrajectoryPoint()
        for i, commanded_joint_name in enumerate(commanded_joint_names):
            error_point.positions.append(clean_named_errors_dict[commanded_joint_name])
            actual_point.positions.append(desired_point.positions[i] - clean_named_errors_dict[commanded_joint_name])

        rospy.logdebug("{0} joint_traj action: sending feedback".format(self.node_name))
        self.feedback.header.stamp = rospy.Time.now()
        self.feedback.joint_names = commanded_joint_names
        self.feedback.desired = desired_point
        self.feedback.actual = actual_point
        self.feedback.error = error_point
        self.server.publish_feedback(self.feedback)

    def success_callback(self, success_str):
        rospy.loginfo("{0} joint_traj action: {1}".format(self.node_name, success_str))
        self.result.error_code = self.result.SUCCESSFUL
        self.result.error_string = success_str
        self.server.set_succeeded(self.result)

if __name__ == "__main__":
    rospy.init_node('stretch_gazebo_adapter')
    adapter = StretchGazeboAdapter(rospy.get_name())
    while not rospy.is_shutdown():
        adapter.loop()
