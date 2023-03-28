#! /usr/bin/env python

import math
import py_trees as pt
import py_trees_ros as ptr

import rospy
import roslaunch
import rospkg
from tf.transformations import euler_from_quaternion

import std_msgs.msg
from geometry_msgs.msg import Twist, Point32
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan, Joy, PointCloud
from nav_msgs.msg import Odometry

import os
import time
import numpy as np
import math
from utils import *

#1 is on robot and 0 is simulation.
run_env=0

class publishLines(pt.behaviour.Behaviour):

    """
    publishes line parameters (to plot lines in rviz)
    """

    def __init__(self, name="publish lines", topic_name="line_end_pts"):
        rospy.loginfo(
            "[PUBLISH LINES] __init__")

        self.topic_name = topic_name
        self.blackboard = pt.blackboard.Blackboard()

        super(publishLines, self).__init__(name)

    def setup(self, timeout):
        """
        Set up things that should be setup only for one time and which generally might 
        require time to prevent delay in tree initialisation
        """
        rospy.loginfo(
            "[PUBLISH LINES] setup")
        self.line_param_pub = rospy.Publisher(
            self.topic_name, PointCloud, queue_size=10)
        self.feedback_message = "setup"
        return True

    def update(self):
        """
        Primary function of the behavior is implemented in this method

        Publishing start and end points of all detected lines in order
        """
        rospy.loginfo(
            "[PUBLISH LINES] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)

        line_parameters = self.blackboard.line_parameters
        point_cld_msg = PointCloud()
        header = std_msgs.msg.Header()
        header.seq = 0
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_laser'  # frame in which the points are measured
        point_cld_msg.header = header

        # appending all end points of all detected lines
        for i in range(len(line_parameters)):
            point_cld_msg.points.append(
                Point32(line_parameters[i][-1][0][0], line_parameters[i][-1][0][1], 0.0))
            point_cld_msg.points.append(
                Point32(line_parameters[i][-1][1][0], line_parameters[i][-1][1][1], 0.0))
        self.line_param_pub.publish(point_cld_msg)
        rospy.loginfo(
            "[PUBLISH LINES] update: SUCCESS, published %0.3f lines" % len(line_parameters))

        return pt.common.Status.SUCCESS


class rotate(pt.behaviour.Behaviour):

    """
    Rotates the robot about z-axis 
    """

    def __init__(self, name="rotate platform", topic_name="/cmd_vel", to_align=False,
                 max_ang_vel=0.1, align_threshold=0.2, wall_priority_thresh=0.6):
        rospy.loginfo("[ROTATE] __init__")

        self.topic_name = topic_name
        self.blackboard = pt.blackboard.Blackboard()
        self.max_ang_vel = max_ang_vel  # units: rad/sec
        self.to_align = to_align
        self.log_start_time = True
        self.align_threshold = align_threshold
        # wall found within 'wall_priority_thresh' distance is prioritised to align with
        self.wall_priority_thresh = wall_priority_thresh
        self.alignment_status = False

        super(rotate, self).__init__(name)

    def setup(self, timeout):
        """
        Set up things that should be setup only for one time and which generally might 
        require time to prevent delay in tree initialisation
        """
        rospy.loginfo("[ROTATE] setup")
        self.cmd_vel_pub = rospy.Publisher(
            self.topic_name, Twist, queue_size=2)
        self.feedback_message = "setup"
        return True

    def initialise(self):
        """
        It will be called the first time your behaviour is ticked and anytime the status is 
        not RUNNING thereafter. 
        """
        rospy.loginfo("[ROTATE] initialise")
        if self.to_align:
            line_parameters = self.blackboard.line_parameters
            # line_parameters: [per_dist, slope, const, end_pts]
            line_parameters.sort(key=lambda x: x[0])
            angle_rad = np.arctan(line_parameters[0][1])
            if abs(line_parameters[0][1]) > 2 and self.blackboard.parent_wall_location == "right":
                if np.all(line_parameters[0][-1][:, 1] < 0):
                    # rotate right
                    angle_rad = -abs(angle_rad)
                else:
                    # rotate left
                    angle_rad = abs(angle_rad)
            if abs(line_parameters[0][1]) > 2 and self.blackboard.parent_wall_location == "left":
                if np.all(line_parameters[0][-1][:, 1] > 0):
                    # rotate left
                    angle_rad = abs(angle_rad)
                else:
                    # rotate right
                    angle_rad = -abs(angle_rad)
            self.angle_to_rotate = angle_rad
            if abs(angle_rad) < self.align_threshold:
                self.alignment_status = True
                self.blackboard.rotation_log = "[initialise] already aligned to wall"
            else:
                self.alignment_status = False
                orientation = self.blackboard.odom_data.orientation
                _, _, self.yaw_at_begining = euler_from_quaternion(
                    [orientation.x, orientation.y, orientation.z, orientation.w])
                self.blackboard.rotation_log = [
                    self.angle_to_rotate, self.yaw_at_begining]

    def update(self):
        """
        Primary function of the behavior is implemented in this method

        Rotating the robot at maximum allowed angular velocity in a given direction, 
        where if _direction_ is +1, it implies clockwise rotation, and if it is -1, it implies
        counter-clockwise rotation
        """
        rospy.loginfo("[ROTATE] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)

        # send the goal
        twist_msg = Twist()
        if self.to_align:
            if self.alignment_status:
                rospy.loginfo("[ROTATE] update: SUCCESS, already aligned")
                self.blackboard.rotation_log = "[update] already aligned to wall"
                return pt.common.Status.SUCCESS

            orientation = self.blackboard.odom_data.orientation
            _, _, current_yaw = euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w])
            twist_msg.angular.z = float(np.sign(
                self.angle_to_rotate)*self.max_ang_vel)

            # as yaw has range b/w +/-2pi, considering the edge cases
            angle_rotated = min(abs(self.yaw_at_begining-current_yaw), -abs(self.yaw_at_begining-current_yaw)%(2*np.pi))
            remaining_angle = abs(self.angle_to_rotate) - angle_rotated            

            if abs(remaining_angle) <= self.align_threshold:
                self.alignment_status = True
                self.blackboard.rotation_log = "[update] aligned successfully"
                return pt.common.Status.SUCCESS
            else:
                self.alignment_status = False
                rospy.loginfo("[ROTATE] update: RUNNING, rotating for [%0.3f radians]" %
                              remaining_angle)

        else:
            rospy.loginfo("[ROTATE] update: RUNNING, rotating for low battery:  [%0.3f]" %
                          self.blackboard.battery)
            twist_msg.angular.z = self.max_ang_vel

        self.cmd_vel_pub.publish(twist_msg)
        return pt.common.Status.RUNNING

    def terminate(self, new_status):
        """
        terminate() is trigerred once the execution of the behavior finishes, 
        i.e. when the status changes from RUNNING to SUCCESS or FAILURE
        """
        rospy.loginfo("[ROTATE] terminate: publishing zero angular velocity")
        twist_msg = Twist()
        twist_msg.angular.z = 0
        self.cmd_vel_pub.publish(twist_msg)
        return super().terminate(new_status)


class move(pt.behaviour.Behaviour):

    """
    To move the robot in a specified direction
    """

    def __init__(self, name="move platform", topic_name="/cmd_vel", towards_wall=False, along_wall=False,
                 avoid_collison=False, direction=[-1.0, 0.0], max_linear_vel=0.1, safe_dist_thresh=0.2, allowance=0.2):
        rospy.loginfo("[MOVE] __init__")

        self.topic_name = topic_name
        self.blackboard = pt.blackboard.Blackboard()
        self.blackboard.parent_wall_location = None
        self.max_linear_vel = max_linear_vel  # units: rad/sec
        self.towards_wall = towards_wall
        self.along_wall = along_wall
        self.avoid_collison = avoid_collison
        self.to_move_bool = False
        self.direction = direction
        self.safe_dist_thresh = safe_dist_thresh
        # allowance: distance allowed to deviate from wall while moving along the wall
        self.allowance = allowance

        super(move, self).__init__(name)

    def setup(self, timeout):
        """
        Set up things that should be setup only for one time and which generally might 
        require time to prevent delay in tree initialisation
        """
        rospy.loginfo("[MOVE] setup")
        self.cmd_vel_pub = rospy.Publisher(
            self.topic_name, Twist, queue_size=2)
        self.feedback_message = "setup"
        return True

    def initialise(self):
        """
        It will be called the first time your behaviour is ticked and anytime the status is 
        not RUNNING thereafter. 
        """
        rospy.loginfo("[MOVE] initialise")

        if self.towards_wall:
            # line_parameters: [per_dist, slope, const, end_pts]
            line_parameters = self.blackboard.line_parameters
            # sorting based on perpendicular distance
            line_parameters.sort(key=lambda x: x[0])

            angle_rad = np.arctan(-1/line_parameters[0][1])
            perpendicular_vector = [np.cos(angle_rad), np.sin(angle_rad)]
            self.direction = [np.sign(-line_parameters[0][2]/line_parameters[0][1])
                              * x for x in perpendicular_vector]  # to specify direction

            # move_log: [distance to move, current location]
            x_at_begining, y_at_begining = self.blackboard.odom_data.position.x, self.blackboard.odom_data.position.y
            self.dist_to_move = line_parameters[0][0]-self.safe_dist_thresh
            self.start_coordinates = np.array([x_at_begining, y_at_begining])
            self.blackboard.move_log = [
                self.dist_to_move, self.start_coordinates]
            rospy.loginfo(
                "[MOVE] initialise(towards_wall): initialised")
            if self.safe_dist_thresh > line_parameters[0][0]:
                self.to_move_bool = False
                self.blackboard.move_log = "[initialise](towards_wall) already closer to wall"
                rospy.loginfo(
                    "[MOVE] initialise(towards_wall): already closer to wall")
            else:
                self.to_move_bool = True

        if self.along_wall:
            # line_parameters: [per_dist, slope, const, end_pts]
            line_parameters = self.blackboard.line_parameters
            # sorting based on perpendicular distance
            line_parameters.sort(key=lambda x: x[0])
            # as it is called after aligning with the wall
            self.direction = [1.0, 0.0]
            if line_parameters[0][2] > 0:
                self.blackboard.parent_wall_location = "left"
            else:
                self.blackboard.parent_wall_location = "right"
            x_at_begining, y_at_begining = self.blackboard.odom_data.position.x, self.blackboard.odom_data.position.y
            self.dist_to_move = (np.linalg.norm(
                line_parameters[0][-1][0]-line_parameters[0][-1][1]))
            self.start_coordinates = np.array([x_at_begining, y_at_begining])
            self.blackboard.move_log = [
                self.dist_to_move, self.start_coordinates]
            rospy.loginfo(
                "[MOVE] initialise(along_wall): initialised")
            self.to_move_bool = True

    def update(self):
        """
        Primary function of the behavior is implemented in update method

        """
        rospy.loginfo("[MOVE] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)

        # send the goal
        twist_msg = Twist()
        if self.along_wall or self.towards_wall:
            if self.blackboard.corner_detected:
                return pt.common.Status.SUCCESS
            elif self.to_move_bool:
                x_curr, y_curr = self.blackboard.odom_data.position.x, self.blackboard.odom_data.position.y
                current_coordinates = np.array([x_curr, y_curr])
                remaining_dist = self.dist_to_move - np.linalg.norm(current_coordinates-self.blackboard.move_log[1])

                if remaining_dist < 0.:
                    self.to_move_bool = False
                    rospy.loginfo(
                        "[MOVE] update: SUCCESS")
                    self.blackboard.move_log = "[update] moved successfully"
                    return pt.common.Status.SUCCESS

                if self.along_wall and self.blackboard.point_at_min_dist>self.safe_dist_thresh+self.allowance:
                    return pt.common.Status.SUCCESS

                twist_msg.linear.x = self.direction[0]*self.max_linear_vel
                twist_msg.linear.y = self.direction[1]*self.max_linear_vel
                rospy.loginfo("[MOVE] update: RUNNING, moving for [%0.3f m]" %
                              remaining_dist)

                self.cmd_vel_pub.publish(twist_msg)
                return pt.common.Status.RUNNING
            rospy.loginfo(
                "[MOVE] update: SUCCESS, already moved")

        elif self.avoid_collison:
            twist_msg.linear.x = self.blackboard.dir_to_avoid_collison[0] * \
                self.max_linear_vel
            twist_msg.linear.y = self.blackboard.dir_to_avoid_collison[1] * \
                self.max_linear_vel
            self.cmd_vel_pub.publish(twist_msg)
            rospy.loginfo("[MOVE] update: RUNNING, moving away from obstacle")
            return pt.common.Status.RUNNING
        return pt.common.Status.SUCCESS

    def terminate(self, new_status):
        """
        terminate() is trigerred once the execution of the behavior finishes, 
        i.e. when the status changes from RUNNING to SUCCESS or FAILURE
        """
        rospy.loginfo("[MOVE] terminate: publishing zero linear velocity")
        twist_msg = Twist()
        if not self.along_wall:
            twist_msg.linear.x = 0
            twist_msg.linear.y = 0
            self.cmd_vel_pub.publish(twist_msg)
        return super().terminate(new_status)


class stopMotion(pt.behaviour.Behaviour):

    """
    Stops the robot when it is controlled using joystick or by cmd_vel command
    """

    def __init__(self, name="stop platform", topic_name1="/cmd_vel", topic_name2="/joy"):
        rospy.loginfo("[STOP MOTION] __init__")

        self.cmd_vel_topic = topic_name1
        self.joy_topic = topic_name2

        super(stopMotion, self).__init__(name)

    def setup(self, timeout):
        """
        Set up things that should be setup only for one time and which generally might 
        require time to prevent delay in tree initialisation
        """
        rospy.loginfo("[STOP MOTION] setup")
        self.cmd_vel_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=2)
        self.joy_pub = rospy.Publisher(self.joy_topic, Joy, queue_size=2)
        self.feedback_message = "setup"
        return True

    def update(self):
        """
        Primary function of the behavior is implemented in this method

        """
        rospy.loginfo("[STOP MOTION] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)

        # send the goal
        twist_msg = Twist()
        twist_msg.linear.x = 0
        twist_msg.linear.y = 0
        self.cmd_vel_pub.publish(twist_msg)

        joy_msg = Joy()
        joy_msg.header.stamp = rospy.Time.now()
        joy_msg.header.frame_id = "/dev/input/js0"
        joy_msg.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joy_msg.buttons = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.joy_pub.publish(joy_msg)

        rospy.loginfo(
            "[STOP MOTION] update: FAILURE, publishing zero cmd_vel and dead-man's switch in joy command")
        return pt.common.Status.FAILURE

    def terminate(self, new_status):
        """
        terminate() is trigerred once the execution of the behavior finishes, 
        i.e. when the status changes from RUNNING to SUCCESS or FAILURE
        """
        rospy.loginfo(
            "[STOP MOTION] terminate: publishing zero angular velocity")
        twist_msg = Twist()
        twist_msg.linear.x = 0
        twist_msg.linear.y = 0
        twist_msg.angular.z = 0

        self.cmd_vel_pub.publish(twist_msg)
        self.sent_goal = False
        return super().terminate(new_status)


class launchNodes(pt.behaviour.Behaviour):
    """
    to run launch files
    """
    def __init__(self, name="LaunchFile", pkg="robile_navigation_demo", launch_file=None, mapping_time_out=300, mapping_dist_thresh=0.2):
        self.name = name
        info = "[LAUNCH "+name+"] __init__"
        rospy.loginfo(info)
        self.blackboard = pt.blackboard.Blackboard()
        self.pkg = pkg
        self.launch_file= launch_file
        self.mapping_time_out = mapping_time_out
        self.mapping_dist_thresh = mapping_dist_thresh
        super(launchNodes, self).__init__(name)

    def setup(self, timeout):
        """
        Set up things that should be setup only for one time and which generally might 
        require time to prevent delay in tree initialisation
        """
        info = "[LAUNCH "+self.name+"] setup"
        rospy.loginfo(info)
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        self.rospack = rospkg.RosPack()
        pkg_path = self.rospack.get_path(self.pkg)
        roslaunch.configure_logging(self.uuid)
        self.path = pkg_path+"/ros/launch/"+self.launch_file
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [self.path])
        self.feedback_message = "setup"
        return True

    def update(self):
        """
        Primary function of the behavior is implemented in this method
        """
        info = "[LAUNCH "+self.name+"] update"
        rospy.loginfo(info)
        if self.launch_file=="gmapping.launch":
            if self.blackboard.gmapping_status=="START":
                self.launch.start()
                self.mapping_start_time = time.time()
                self.blackboard.gmapping_status = "RUN"
                info = "[LAUNCH "+self.name+"] update: started"
                rospy.loginfo(info)
                return pt.common.Status.RUNNING
            elif self.blackboard.gmapping_status=="RUN":
                # current_coordinate = np.array([self.blackboard.odom_data.position.x, self.blackboard.odom_data.position.y])
                # dist_from_start_coord = np.linalg.norm(self.blackboard.start_coordinate-current_coordinate)
                current_time = time.time()
                time_passed = current_time-self.mapping_start_time
                if time_passed > self.mapping_time_out:
                    self.blackboard.gmapping_status = "SAVE"
                    return pt.common.Status.SUCCESS
                return pt.common.Status.RUNNING

        if self.launch_file=="amcl.launch":
            if self.blackboard.amcl_status=="START":
                self.launch.start()
                self.blackboard.amcl_status = "RUN"
                info = "[LAUNCH "+self.name+"] update: started"
                rospy.loginfo(info)

        if self.launch_file=="move_base_dwa.launch":
            if self.blackboard.move_base_dwa_status=="START":
                self.launch.start()
                self.blackboard.move_base_dwa_status = "RUN"
                info = "[LAUNCH "+self.name+"] update: started"
                rospy.loginfo(info)
                return pt.common.Status.RUNNING
            elif self.blackboard.move_base_dwa_status == "RUN":
                return pt.common.Status.RUNNING
        return pt.common.Status.SUCCESS


class saveMap(pt.behaviour.Behaviour):
    """
    save and load the saved map after 
    """
    def __init__(self, name="SaveMap", map_name="sim_env"):
        rospy.loginfo("[SAVE MAP] __init__")
        self.map_saved = False
        self.map_name = map_name
        self.blackboard = pt.blackboard.Blackboard()
        self.rospack = rospkg.RosPack()
        self.map_path = self.rospack.get_path('robile_default_env_config')+'/ros/maps/'+self.map_name
        super(saveMap, self).__init__(name)

    def update(self):
        """
        Primary function of the behavior is implemented in this method
        """
        rospy.loginfo("[SAVE MAP] update")
        if (not self.map_saved) and self.blackboard.gmapping_status=='SAVE':
            node_description = "rosrun map_server map_saver -f "+self.map_path+" --free 20 --occ 80"
            os.system(node_description)
            os.system('rosnode kill /slam_gmapping')
            os.environ['ROBOT_ENV'] = "maps/" + self.map_name
            self.map_saved = True
            self.blackboard.amcl_status = 'START' 
            self.blackboard.move_base_dwa_status = 'START'
        return pt.common.Status.SUCCESS 


class batteryStatus2bb(ptr.subscribers.ToBlackboard):

    """
    Checking battery status
    """

    def __init__(self,  name, topic_name="/mileage", threshold=30.0):
        rospy.loginfo("[BATTERY] __init__")
        super(batteryStatus2bb, self).__init__(name=name,
                                                topic_name=topic_name,
                                                topic_type=Float32,
                                                blackboard_variables={
                                                    'battery': 'data'},
                                                initialise_variables={
                                                    'battery': 0.0},
                                                clearing_policy=pt.common.ClearingPolicy.NEVER
                                                )
        self.blackboard = pt.blackboard.Blackboard()
        self.blackboard.battery_low_warning = False
        self.threshold = threshold
       

    def update(self):
        """
        Primary function of the behavior is implemented in this method

        Call the parent to write the raw data to the blackboard and then check against the
        threshold to determine if the low warning flag should also be updated.
        """
        rospy.loginfo('[BATTERY] update')
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(batteryStatus2bb, self).update()

        if status != pt.common.Status.RUNNING:
            if self.blackboard.battery < self.threshold:
                rospy.loginfo('[BATTERY] update: lesser than threshold')
                self.blackboard.battery_low_warning = True
                rospy.logwarn_throttle(
                    30, "%s: battery level is low!" % self.name)
            else:
                rospy.loginfo('[BATTERY] update: greater than threshold')
                self.blackboard.battery_low_warning = False

            self.feedback_message = "Battery level is low" if self.blackboard.battery_low_warning else "Battery level is ok"
        return status


class laserScanFiltered2bb(ptr.subscribers.ToBlackboard):

    """
    Checking filtered laser_scan to avoid possible collison
    """

    def __init__(self, name, topic_name="/scan_filtered", safe_range=0.4):
        rospy.loginfo("[LASER SCAN] __init__")
        super(laserScanFiltered2bb, self).__init__(name=name,
                                                     topic_name=topic_name,
                                                     topic_type=LaserScan,
                                                     blackboard_variables={
                                                         'laser_scan': 'ranges', 'max_angle': 'angle_max', 'min_angle': 'angle_min',
                                                         'max_range': 'range_max', 'min_range': 'range_min'},
                                                     # to dictate when data should be cleared/reset.
                                                     clearing_policy=pt.common.ClearingPolicy.NEVER
                                                     )
        self.blackboard = pt.blackboard.Blackboard()
        self.blackboard.collison_warning = False
        self.safe_min_range = safe_range
        self.blackboard.point_at_min_dist = 0.0

    def update(self):
        """
        Primary function of the behavior is implemented in this method

        Call the parent to write the raw data to the blackboard and then check against the
        threshold to set the warning if the robot is close to any obstacle.
        """
        global run_env
        rospy.loginfo(
            "[LASER SCAN] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(laserScanFiltered2bb, self).update()

        if status != pt.common.Status.RUNNING:
            # splitting data into segments to handle possible noises
            segment_size = 20
            scan_data_np_array= np.array(self.blackboard.laser_scan)
            ##
            ##
            # ONLY FOR RUNNING ON ROBOT
            ##
            ##
            if run_env==1:
                scan_data_np_array = np.append(scan_data_np_array,0.0)
            self.blackboard.laser_scan = "Processed(laserScanFiltered2bb)"
            # laser scanner points which are outside the min and max range are assigned zero distance
            scan_data_np_array[scan_data_np_array == 0.0] = 20.0
            scan_data_np_array = scan_data_np_array.reshape(segment_size, -1)
            # scan_data_np_array = np.array_split(scan_data_np_array,len(scan_data_np_array)/segment_size)
            scan_data_segment_mean = np.mean(scan_data_np_array, axis=1)
            self.blackboard.point_at_min_dist = np.min(scan_data_segment_mean)

            if self.blackboard.point_at_min_dist < self.safe_min_range:
                rospy.loginfo("[LASER SCAN] update: possible collison detected at [%0.3f meters]" %
                              self.blackboard.point_at_min_dist)
                self.blackboard.collison_warning = True
                segment_ang_step = (
                    self.blackboard.max_angle-self.blackboard.min_angle)/segment_size
                dir_ang_rad = np.pi+self.blackboard.min_angle + \
                    ((np.argmin(scan_data_segment_mean)+0.5))*(segment_ang_step)
                self.blackboard.dir_to_avoid_collison = [
                    np.cos(dir_ang_rad), np.sin(dir_ang_rad)]

                rospy.logwarn_throttle(
                    30, "%s: possible collison detected!" % self.name)
            else:
                rospy.loginfo("[LASER SCAN] update: no collison detected")
                self.blackboard.collison_warning = False

            self.feedback_message = "Possible collison detected!" if self.blackboard.collison_warning else "Collison status: free to move"

        return status


class odom2bb(ptr.subscribers.ToBlackboard):

    """
    Saving odometry data in blackboard
    """

    def __init__(self, name, topic_name="/odom"):
        rospy.loginfo("[ODOM] __init__")
        super(odom2bb, self).__init__(name=name,
                                      topic_name=topic_name,
                                      topic_type=Odometry,
                                      blackboard_variables={
                                          'odom_data': 'pose.pose'},
                                      clearing_policy=pt.common.ClearingPolicy.NEVER
                                      )

        self.blackboard = pt.blackboard.Blackboard()
        self.save_initial_pose = True
        # initialising navigation launch file status
        self.blackboard.gmapping_status = 'STOP'
        self.blackboard.amcl_status = 'STOP'
        self.blackboard.move_base_dwa_status = 'STOP'        

    def update(self):
        """
        Collecting odometry data of the robot
        """
        rospy.loginfo("[ODOM] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status_subscribe = super(odom2bb, self).update()
        if self.save_initial_pose:
            # self.blackboard.start_coordinate = np.array([self.blackboard.odom_data.position.x, self.blackboard.odom_data.position.y])
            self.blackboard.gmapping_status = 'START'
            self.save_initial_pose = False
        return status_subscribe


class gridOccupancyStatus2bb(ptr.subscribers.ToBlackboard):

    """
    Determining whether the grid consists of sufficient laser scan points
    """

    def __init__(self, name, topic_name="/scan_filtered", min_points=100, grid_size_x=1, grid_size_y=1.4):
        rospy.loginfo("[GRID STATUS] __init__")
        super(gridOccupancyStatus2bb, self).__init__(name=name,
                                                       topic_name=topic_name,
                                                       topic_type=LaserScan,
                                                       blackboard_variables={
                                                           'laser_scan_grid': 'ranges', 'max_angle': 'angle_max', 'min_angle': 'angle_min'},
                                                       clearing_policy=pt.common.ClearingPolicy.NEVER
                                                       )
        self.blackboard = pt.blackboard.Blackboard()
        self.blackboard.grid_occupied = False
        self.blackboard.grid_occupancy_count = 0
        # minimum points to consider the grid to be filled for wall detection
        self.min_points = min_points
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

    def update(self):
        """
        Collecting parameters of line associated with each wall
        """
        rospy.loginfo(
            "[GRID STATUS] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(gridOccupancyStatus2bb, self).update()

        if status != pt.common.Status.RUNNING:
            laser_scan_array = np.array(self.blackboard.laser_scan_grid)
            self.blackboard.laser_scan_grid = "Processed(gridOccupancyStatus2bb)"
            angles = np.linspace(
                self.blackboard.min_angle, self.blackboard.max_angle, laser_scan_array.shape[0])
            laser_sub_cartesian = np_polar2cart(
                np.array([laser_scan_array, angles]).T)
            inliers = abs(laser_sub_cartesian) < np.array(
                [self.grid_size_x, self.grid_size_y/2])
            inlier_count = inliers[~np.any(
                inliers == False, axis=1), :].shape[0]

            self.blackboard.grid_occupancy_count = inlier_count
            if inlier_count > self.min_points:
                self.blackboard.grid_occupied = True
            else:
                self.blackboard.grid_occupied = False
            rospy.loginfo(
                "[GRID STATUS] update: number of points found are [%0.3f]" % inlier_count)
            self.feedback_message = "Sufficient points detected in the grid" if inlier_count > self.min_points else "Not sufficient points detected in the grid"

        return status


class wallParamRANSAC2bb(ptr.subscribers.ToBlackboard):

    """
    Collecting parameters of line associated with each wall using RANSAC on scan filtered data
    """

    def __init__(self, name, topic_name="/scan_filtered", dist_thresh=0.1, iterations=15, thresh_count=15, sigma=1, rf_max_pts=5, wall_priority_thresh=0.8):
        rospy.loginfo(
            "[WALL PARAMETERS RANSAC] __init__")
        super(wallParamRANSAC2bb, self).__init__(name=name,
                                                   topic_name=topic_name,
                                                   topic_type=LaserScan,
                                                   blackboard_variables={
                                                       'laser_scan_ransac': 'ranges', 'max_angle': 'angle_max', 'min_angle': 'angle_min',
                                                       'max_range': 'range_max', 'min_range': 'range_min'},
                                                   clearing_policy=pt.common.ClearingPolicy.NEVER
                                                   )
        self.blackboard = pt.blackboard.Blackboard()
        # The parameter used to determine the furthest a point can be from the line and still be considered an inlier
        self.dist_thresh = dist_thresh
        # k: The number of iterations the RANSAC algorithm will run for
        self.iterations = iterations
        # number of minimum points required to form a line
        self.thresh_count = thresh_count
        # maximum number of points allowed in a cluster (used in reduction filter)
        self.rf_max_pts = rf_max_pts
        # maximum distance between two consecutive points to consider them (used in reduction filter)
        self.sigma = sigma
        # wall found within 'wall_priority_thresh' distance is prioritised to align with
        self.wall_priority_thresh = wall_priority_thresh
        self.blackboard.corner_detected = False        
        self.blackboard.aligned = False

    def update(self):
        """
        Collecting parameters of line associated with each wall
        """
        rospy.loginfo(
            "[WALL PARAMETERS RANSAC] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status_subscribe = super(wallParamRANSAC2bb, self).update()
        if status_subscribe != pt.common.Status.RUNNING:
            # if self.blackboard.grid_occupied:
            #     rospy.loginfo(
            #         '[WALL PARAMETERS RANSAC] update: SUCCESS, grids are occupied, grid occupancy status will be used')
            #     self.blackboard.laser_scan_ransac = "Occupied grid cells are considered"
            #     return pt.common.Status.FAILURE
            #     #  return status_subscribe
            # else:
            self.blackboard.line_parameters = []
            laser_sub_cartesian = process_data(range_data=np.array(self.blackboard.laser_scan_ransac), max_angle=self.blackboard.max_angle,
                                                min_angle=self.blackboard.min_angle, max_range=self.blackboard.max_range, min_range=self.blackboard.min_range,
                                                sigma=self.sigma, rf_max_pts=self.rf_max_pts, reduce_bool=True)
            self.blackboard.laser_scan_ransac = "Processed(wallParamRANSAC2bb)"
            points = [point for point in laser_sub_cartesian]
            self.blackboard.line_parameters = RANSAC_get_line_params(
                points, self.dist_thresh, self.iterations, self.thresh_count)
            # detecting the corner and discarding parent wall
            line_parameters = self.blackboard.line_parameters
            line_parameters.sort(key=lambda x: x[0])            
            for line_param in line_parameters:
                if abs(line_param[1]) > 2 and line_param[0] < self.wall_priority_thresh:
                    self.blackboard.line_parameters = [line_param]
                    self.blackboard.corner_detected = True
                    rospy.loginfo(
                        "[ROTATE] initialise: Corner detected")
                    break
                self.blackboard.corner_detected = False

            self.blackboard.num_walls = len(
                self.blackboard.line_parameters)

            rospy.loginfo("[WALL PARAMETERS RANSAC] update: SUCCESS,number of walls found are [%0.3f]" % len(
                self.blackboard.line_parameters))
            self.feedback_message = "Line parameters collection successful (RANSAC)" if self.blackboard.line_parameters else "Line parameters not updated (RANSAC)"

        return status_subscribe


class wallParamGrid2bb(ptr.subscribers.ToBlackboard):

    """
    Collecting parameters of line associated with each wall using occupied cells from grid, by optionally using RANSAC or online line extraction algorithm
    """

    def __init__(self, name, topic_name="/scan_filtered", grid_size_x=1, grid_size_y=1.4, grid_width=0.04,
                 occupancy_thresh=0.1, allowed_deviation=0.45, incr=0.01, max_dist=0.1, min_points=5,
                 sigma=1, rf_max_pts=5, iterations=10, algorithm="ransac", wall_priority_thresh=0.8):
        rospy.loginfo(
            "[WALL PARAMETERS Grid] __init__")
        super(wallParamGrid2bb, self).__init__(name=name,
                                                 topic_name=topic_name,
                                                 topic_type=LaserScan,
                                                 blackboard_variables={
                                                     'laser_scan_grid': 'ranges', 'max_angle': 'angle_max',
                                                     'min_angle': 'angle_min', 'max_range': 'range_max', 'min_range': 'range_min'},
                                                 clearing_policy=pt.common.ClearingPolicy.NEVER
                                                 )

        self.blackboard = pt.blackboard.Blackboard()
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid_width = grid_width
        self.algorithm = algorithm

        # assigned to an array of probability of occupancy of individual grid cells
        self.occupancy_thresh = occupancy_thresh

        # allowed fraction of deviation of sum of distances between consecutive points and
        # (Online) the total length  of the line
        self.e = allowed_deviation
        # (Online) increment in the error values with the number of points
        self.incr = incr
        # (Online) maximum distance between consecutive points allowed in a line segment
        self.max_dist = max_dist
        # (Online) minimum number of points required in a line segment
        self.k = min_points
        # (RANSAC) The parameter used to determine the furthest a point can be from the line and still be considered an inlier
        self.dist_thresh = 3*grid_width
        # (RANSAC) The number of iterations the RANSAC algorithm will run for
        self.iterations = iterations
        # (RANSAC) number of minimum points required to form a line
        self.thresh_count = min_points*2
        # maximum number of points allowed in a cluster (used in reduction filter)
        self.rf_max_pts = rf_max_pts
        # maximum distance between two consecutive points to consider them (used in reduction filter)
        self.sigma = sigma
        # wall found within 'wall_priority_thresh' distance is prioritised to align with
        self.wall_priority_thresh = wall_priority_thresh
        self.blackboard.corner_detected = False

    def update(self):
        """
        Collecting parameters of line associated with each wall
        """
        rospy.loginfo(
            "[WALL PARAMETERS Grid] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status_subscribe = super(wallParamGrid2bb, self).update()

        if status_subscribe != pt.common.Status.RUNNING:

            if self.blackboard.grid_occupied:
                self.blackboard.line_parameters = []

                # TEST: use raw data in cartesian form instead of using reduced data
                laser_sub_cartesian = process_data(range_data=np.array(self.blackboard.laser_scan_grid), max_angle=self.blackboard.max_angle,
                                                   min_angle=self.blackboard.min_angle, max_range=self.blackboard.max_range, min_range=self.blackboard.min_range,
                                                   sigma=self.sigma, rf_max_pts=self.rf_max_pts, reduce_bool=False)

                # get probability of cell being occupied
                grid_occupancy_probability = np.round(get_occupancy_probability(
                    laser_sub_cartesian, self.grid_size_x, self.grid_size_y, self.grid_width), 2)

                # get coordinates of occupied cells
                array_of_occupied_cells = get_array_of_occupied_cells(
                    grid_occupancy_probability, self.occupancy_thresh, self.grid_size_x, self.grid_size_y, self.grid_width)

                if self.algorithm == 'online':
                    # get line parameters from the occupied cell coordinates
                    self.blackboard.laser_scan_grid = "Processed(wallParamGrid2bb)-Online"
                    self.blackboard.line_parameters = online_get_line_params(
                        array_of_occupied_cells, self.e, self.incr, self.max_dist, self.k)
                if self.algorithm == 'ransac':
                    self.blackboard.laser_scan_grid = "Processed(wallParamGrid2bb)-RANSAC"
                    rospy.loginfo(
                        "[WALL PARAMETERS Grid] update: using RANSAC")
                    # get line parameters from the occupied cell coordinates
                    points = [point for point in array_of_occupied_cells]
                    self.blackboard.line_parameters = RANSAC_get_line_params(
                        points, self.dist_thresh, self.iterations, self.thresh_count)
                    if len(self.blackboard.line_parameters) == 0:
                        self.blackboard.laser_scan_grid = "Processed(wallParamGrid2bb)-Online"
                        rospy.loginfo(
                            "[WALL PARAMETERS Grid] update: using Online line detection")
                        # get line parameters from the occupied cell coordinates
                        self.blackboard.line_parameters = online_get_line_params(
                            array_of_occupied_cells, self.e, self.incr, self.max_dist, self.k)

                # detecting the corner and discarding parent wall
                line_parameters = self.blackboard.line_parameters
                line_parameters.sort(key=lambda x: x[0])                
                for line_param in line_parameters:
                    if abs(line_param[1]) > 2 and line_param[0] < self.wall_priority_thresh:
                        self.blackboard.line_parameters = [line_param]
                        self.blackboard.corner_detected = True
                        break
                    self.blackboard.corner_detected = False

                self.blackboard.num_walls = len(
                    self.blackboard.line_parameters)

                rospy.loginfo("[WALL PARAMETERS Grid] update: SUCCESS, number of walls found are [%0.3f]" % len(
                    self.blackboard.line_parameters))
                self.feedback_message = "Line parameters collection successful" if self.blackboard.line_parameters else "Line parameters not updated"
                return pt.common.Status.SUCCESS
            else:
                self.blackboard.laser_scan_grid = "RANSAC is being used"
                rospy.loginfo(
                    '[WALL PARAMETERS Grid] update: SUCCESS, grids NOT occupied, using RANSAC')
                return pt.common.Status.FAILURE

        else:
            return status_subscribe
        

class rotate_dummy(pt.behaviour.Behaviour):

    """
    Rotates the robot about z-axis 
    """

    def __init__(self, name="rotate platform", topic_name="/cmd_vel", to_align=False,
                 max_ang_vel=0.1, align_threshold=0.2, wall_priority_thresh=0.6):
        rospy.loginfo("[ROTATE_DUMMY] __init__")

        self.topic_name = topic_name
        self.blackboard = pt.blackboard.Blackboard()
        # self.max_ang_vel = max_ang_vel  # units: rad/sec
        # self.to_align = to_align
        # self.log_start_time = True
        # self.align_threshold = align_threshold
        # wall found within 'wall_priority_thresh' distance is prioritised to align with
        # self.wall_priority_thresh = wall_priority_thresh
        # self.alignment_status = False
        # compare
        self.lin_vel = 0.2
        self.rot_vel = 0.1
        self.blackboard = pt.blackboard.Blackboard()
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.kP = 0.05# 0.5 maybe
        # Moving this line to initialise
        # self.aligned = False
        self.cmd_vel_pub = rospy.Publisher(
            self.topic_name, Twist, queue_size=2)
        self.r = rospy.Rate(10)
        self.parallel_to_wall_diff = 0.0
        # self.pub = rospy.Publisher(self.topic_name, Twist, queue_size=10)
        # self.pub_fb = rospy.Publisher('feedback', Bool, queue_size=10)
        self.sub = rospy.Subscriber("/scan_filtered", LaserScan, self.scanCB)
        
        self.is_looking_to_wall = False
        self.stop_aligning_flag = False # TODO: comment this line in future and use blackboard variable
        # self.blackboard.aligned = False
        self.angle_rad = None
        super(rotate_dummy, self).__init__(name)

    def setup(self, timeout):
        """
        Set up things that should be setup only for one time and which generally might 
        require time to prevent delay in tree initialisation
        """
        rospy.loginfo("[ROTATE_DUMMY] setup")
        # self.cmd_vel_pub = rospy.Publisher(
        #     self.topic_name, Twist, queue_size=2)
        self.cmd_vel_pub = rospy.Publisher(
            self.topic_name, Twist, queue_size=2)
        self.feedback_message = "setup"
        return True

    def initialise(self):
        """
        It will be called the first time your behaviour is ticked and anytime the status is 
        not RUNNING thereafter. 
        """
        
        rospy.loginfo("[ROTATE_DUMMY] initialise")
        self.aligned = False
        self.blackboard.aligned = False
        line_params = self.blackboard.line_parameters
        line_params.sort(key=lambda x: x[0])
        angle_rad = np.arctan(-1/line_params[0][1])
        orientation_q = self.blackboard.odom_data.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.yaw) = euler_from_quaternion (orientation_list)
        # if self.yaw<0:
        #     self.yaw+=6.28319
        self.initial_angle = self.yaw
        if line_params[0][2] > 0:
            self.blackboard.parent_wall_location = "left"
        else:
            self.blackboard.parent_wall_location = "right"
        if abs(line_params[0][1]) > 2 and self.blackboard.parent_wall_location == "right":
            if np.all(line_params[0][-1][:, 1] < 0):
                # rotate right
                angle_rad = -abs(angle_rad)
            else:
                # rotate left
                angle_rad = abs(angle_rad)
        if abs(line_params[0][1]) > 2 and self.blackboard.parent_wall_location == "left":
            if np.all(line_params[0][-1][:, 1] > 0):
                # rotate left
                angle_rad = abs(angle_rad)
            else:
                # rotate right
                angle_rad = -abs(angle_rad)
        self.angle_rad = angle_rad
        self.rotate_angle = self.yaw + self.angle_rad
        if self.rotate_angle > math.pi:
            self.rotate_angle -=2*math.pi
        elif self.rotate_angle < -math.pi:
           self.rotate_angle +=2*math.pi
        
        self.target_angle = self.rotate_angle + 1.47
        if self.target_angle > math.pi:
            self.target_angle-=2*math.pi
        elif self.target_angle < -math.pi:
            self.target_angle +=2*math.pi

    def update(self):
        """
        Primary function of the behavior is implemented in this method

        Rotating the robot at maximum allowed angular velocity in a given direction, 
        where if _direction_ is +1, it implies clockwise rotation, and if it is -1, it implies
        counter-clockwise rotation
        """
        self.blackboard.zDateAlignToWall = time.time()
        rospy.loginfo("[ROTATE_DUMMY] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)
        orientation_q = self.blackboard.odom_data.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.yaw) = euler_from_quaternion (orientation_list)


        # if self.yaw<0:
        #     self.yaw+=6.28319
        self.blackboard.zpresent_Yaw_= self.yaw
        self.blackboard.ztarget_angle=self.target_angle
        
        self.angle_diff = self.rotate_angle - self.yaw
        if self.angle_diff > math.pi:
            self.angle_diff -=2*math.pi
        elif self.angle_diff < -math.pi:
           self.angle_diff +=2*math.pi
        self.blackboard.zrotate_angle= self.rotate_angle
        if not self.aligned:
        
            # if (self.lsrData[self.mid_idx]>=min(self.lsrData)+0.02 or self.lsrData[self.mid_idx] == 20.0) and not (self.ranges.index(min(self.ranges))<self.mid_idx+10 and self.ranges.index(min(self.ranges))>self.mid_idx-10):

            # if not -0.2<=self.angle_rad-self.yaw<= 0.2:
            # if not abs(abs(self.initial_angle)-abs(self.yaw)) >= abs(self.angle_rad):
            self.blackboard.zzAngDiff = self.angle_diff
            self.blackboard.zzYaw= self.yaw
            self.blackboard.zzRotateAngle = self.rotate_angle
            if abs(self.angle_diff) > 0.4:
            # if abs(self.initial_angle-self.yaw) <= abs(self.angle_rad)+0.2 or abs(self.initial_angle-self.yaw) >= abs(self.angle_rad)-0.2:
                self.look_at_wall()
            elif self.lsrData[self.mid_idx] < 0.5:  
                self.aligned = True           
                rospy.loginfo("stop")
                
                self.stop_robot()
                # self.pub_fb.publish(True)
                self.is_looking_to_wall = True
                self.wall_function()
                return pt.common.Status.SUCCESS
            else:
                self.move_forward()

        if self.stop_aligning_flag:
            return pt.common.Status.SUCCESS
        
        return pt.common.Status.RUNNING
        

    def terminate(self, new_status):
        """
        terminate() is trigerred once the execution of the behavior finishes, 
        i.e. when the status changes from RUNNING to SUCCESS or FAILURE
        """
        rospy.loginfo("[ROTATE_DUMMY] terminate: publishing zero angular velocity")
        twist_msg = Twist()
        twist_msg.angular.z = 0
        self.cmd_vel_pub.publish(twist_msg)
        rospy.logwarn(pt.common.Status)
        return super().terminate(new_status)

    def scanCB(self, data_laser):
        # rospy.loginfo("[ROTATE_DUMMY] ScanCB")
        data= np.array(data_laser.ranges)
        if run_env==1:
            data = np.append(data,0.0)
        # laser scanner points which are outside the min and max range are assigned zero distance
        data[data == 0.0] = 20.0
        self.ranges = list(data)
        self.lsrData = self.ranges 
        self.ranges = [20.0 if x==0.0 else x for x in self.ranges] 
        self.lsrData = self.ranges
        self.lsrData = [20.0 if x==0.0 else x for x in self.lsrData]
        self.mid_idx = int(len(self.lsrData)/2)
        # if not self.aligned:
        #     if (self.lsrData[self.mid_idx]>=min(self.lsrData)+0.005 or self.lsrData[self.mid_idx] == math.inf) and not (self.ranges.index(min(self.ranges))<self.mid_idx+2 and self.ranges.index(min(self.ranges))>self.mid_idx-2):

        #         self.look_at_wall()
        #     elif self.lsrData[self.mid_idx] < 0.3:  
        #         self.aligned = True           
        #         rospy.loginfo("stop")
                
        #         self.stop_robot()
        #         # self.pub_fb.publish(True)
        #         self.is_looking_to_wall = True
        #         self.wall_function()
        #         return pt.common.Status.SUCCESS
        #     else:
        #         self.move_forward()
        #         return pt.common.Status.RUNNING

    def look_at_wall(self):
        # if self.lsrData[self.mid_idx+5]>self.lsrData[self.mid_idx-5] or self.lsrData[self.mid_idx] == math.inf:
        # if self.lsrData[self.mid_idx+5]>self.lsrData[self.mid_idx-5] or self.lsrData[self.mid_idx] == math.inf:

        # if self.angle_rad>0:
        if self.angle_diff < 0:
            rospy.loginfo("right")
            self.rotate_right()
        # elif self.lsrData[self.mid_idx+5]<self.lsrData[self.mid_idx-5]:
        # elif self.angle_rad<0 :
        elif self.angle_diff > 0:
            rospy.loginfo("left")
            self.rotate_left()

    def rotate_right(self):
        tMsg = Twist()
        tMsg.angular.z = -float(self.rot_vel)
        self.cmd_vel_pub.publish(tMsg)

    def rotate_left(self):
        tMsg = Twist()
        tMsg.angular.z = float(self.rot_vel)
        self.cmd_vel_pub.publish(tMsg)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())
        # return pt.common.Status.RUNNING
    
    def move_forward(self):
        tMsg = Twist()
        tMsg.linear.x = 0.1
        # rospy.loginfo("forward")
        self.cmd_vel_pub.publish(tMsg)

    def wall_function(self):
        if self.is_looking_to_wall == True:
            print('Starting to read Odom data')
            if self.stop_aligning_flag == False:
                sub = rospy.Subscriber ('/odom', Odometry, self.get_rotation)
                self.align_to_wall()

    def get_rotation (self, msg):
        # global roll, pitch, yaw
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion (orientation_list)
        rospy.loginfo(self.yaw)
        self.parallel_to_wall_diff = self.target_angle - self.yaw
        if self.parallel_to_wall_diff  > math.pi:
            self.parallel_to_wall_diff  -=2*math.pi
        elif self.parallel_to_wall_diff  < -math.pi:
           self.parallel_to_wall_diff  +=2*math.pi
        self.blackboard.zDiff = self.parallel_to_wall_diff 
        if abs(self.parallel_to_wall_diff ) > 0.03: #self.target_angle * math.pi/180: #TODO Stop overshooting! Trial and error instead of 1.51.
            self.stop_aligning_flag = True
            self.blackboard.aligned = self.stop_aligning_flag

    def align_to_wall(self):
        # pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        command= Twist()
        r = rospy.Rate(10)
        while not self.stop_aligning_flag:
            print(self.stop_aligning_flag)
            command.angular.z = self.kP  * (self.parallel_to_wall_diff)
            self.cmd_vel_pub.publish(command)
            r.sleep()
        self.stop_robot()

class move_dummy(pt.behaviour.Behaviour):

    """
    To move the robot in a specified direction
    """

    def __init__(self, name="move platform", topic_name="/cmd_vel", towards_wall=False, along_wall=False,
                 avoid_collison=False, direction=[-1.0, 0.0], max_linear_vel=0.1, safe_dist_thresh=0.2, allowance=0.2):
        rospy.loginfo("[MOVE_DUMMY] __init__")

        self.topic_name = topic_name
        self.blackboard = pt.blackboard.Blackboard()
        self.blackboard.timeup = False
        # self.blackboard.parent_wall_location = None
        # self.max_linear_vel = max_linear_vel  # units: rad/sec
        # self.towards_wall = towards_wall
        # self.along_wall = along_wall
        # self.avoid_collison = avoid_collison
        # self.to_move_bool = False
        # self.direction = direction
        # self.safe_dist_thresh = safe_dist_thresh
        # # allowance: distance allowed to deviate from wall while moving along the wall
        # self.allowance = allowance
        self.object_name = name
        self.hz = 10                     # Cycle Frequency
        self.loop_index = 0              # Number of sampling cycles
        self.loop_index_outer_corner = 0 # Loop index when the outer corner is detected
        self.loop_index_inner_corner = 0 # Loop index when the inner corner is detected
        self.inf = 20                     # Limit to Laser sensor range in meters, all distances above this value are 
        self.wall_dist = 0.5           # Distance desired from the wall
        self.max_speed = 0.2             # Maximum speed of the robot on meters/seconds
        self.p = 0.05                      # Proportional constant for controller  
        self.d = 0.05                       # Derivative constant for controller 
        self.angle = 0.15                   # Proportional constant for angle controller (just simple P controller)

        self.direction = -1              # 1 for wall on the left side of the robot (-1 for the right side)
        self.e = 0                       # Diference between current wall measurements and previous one
        self.angle_min = 0               # Angle, at which was measured the shortest distance between the robot and a wall
        self.dist_front = 0              # Measured front distance
        self.diff_e = 0                  # Difference between current error and previous one
        self.dist_min = 0                # Minimum measured distance
        # Time when the last outer corner; direction and inner corner were detected or changed.
        self.last_outer_corner_detection_time = time.time()
        self.last_change_direction_time = time.time()
        self.last_inner_corner_detection_time = time.time()
        self.rotating = 0 
        # self.pub_ = None

        # Sensor regions
        self.regions = {
                'bright': 0,
                'right': 0,
                'fright': 0,
                'front': 0,
                'left': 0,
        }
        self.last_kinds_of_wall=[0, 0, 0, 0, 0]
        self.index = 0
        self.state_outer_inner=[0, 0, 0, 0]
        self.index_state_outer_inner = 0
        self.bool_outer_corner = 0
        self.bool_inner_corner =0
        self.last_vel = [random.uniform(0.1,0.3),  random.uniform(-0.3,0.3)]
        self.wall_found =0

        #Robot state machines
        self.state = 0
        self.state_dict = {
            0: 'random wandering',
            1: 'following wall',
            2: 'rotating',
            3: 'obstacle avoidance'
        }

        self.pub = rospy.Publisher(self.topic_name, Twist, queue_size=10)
    
        self.sub = rospy.Subscriber('/scan_filtered', LaserScan, self.clbk_laser)

        self.rate = rospy.Rate(self.hz)

        print('%s is running' %(self.object_name))

        super(move_dummy, self).__init__(name)

    def setup(self, timeout):
        """
        Set up things that should be setup only for one time and which generally might 
        require time to prevent delay in tree initialisation
        """
        rospy.loginfo("[MOVE_DUMMY] setup")
        self.cmd_vel_pub = rospy.Publisher(
            self.topic_name, Twist, queue_size=2)
        self.feedback_message = "setup"
        return True

    def initialise(self):
        """
        It will be called the first time your behaviour is ticked and anytime the status is 
        not RUNNING thereafter. 
        
        """

        rospy.loginfo("[MOVE_DUMMY] initialise")


    def update(self):
        """
        Primary function of the behavior is implemented in update method

        """
        rospy.loginfo("[MOVE_DUMMY] update")
        self.logger.debug("%s.update()" % self.__class__.__name__)

        if not self.blackboard.gmapping_status == "SAVE":
            self.loop_index = self.loop_index + 1
            self.msg = Twist()

            # State Dispatcher
            if self.state == 0:
                self.msg = self.random_wandering()
                self.pub.publish(self.msg)
            elif self.state == 1:
                self.msg = self.following_wall()
                print('%s is following wall' %(self.object_name))
                self.pub.publish(self.msg)
            elif self.state == 2:
                print('%s is turning' %(self.object_name))
                self.msg = self.rotating()
                self.pub.publish(self.msg)
            elif self.state == 3:
                self.obstacle_avoidance()
                # self.pub_.publish(ooo)
            else:
                rospy.logerr('Unknown state!')  

            self.rate.sleep()
        else:
            self.blackboard.timeup=True
            return pt.common.Status.SUCCESS
        
        return pt.common.Status.RUNNING
        # send the goal
        # twist_msg = Twist()
        # if self.along_wall or self.towards_wall:
        #     if self.blackboard.corner_detected:
        #         return pt.common.Status.SUCCESS
        #     elif self.to_move_bool:
        #         x_curr, y_curr = self.blackboard.odom_data.position.x, self.blackboard.odom_data.position.y
        #         current_coordinates = np.array([x_curr, y_curr])
        #         remaining_dist = self.dist_to_move - np.linalg.norm(current_coordinates-self.blackboard.move_log[1])

        #         if remaining_dist < 0.:
        #             self.to_move_bool = False
        #             rospy.loginfo(
        #                 "[MOVE_DUMMY] update: SUCCESS")
        #             self.blackboard.move_log = "[update] moved successfully"
        #             return pt.common.Status.SUCCESS

        #         if self.along_wall and self.blackboard.point_at_min_dist>self.safe_dist_thresh+self.allowance:
        #             return pt.common.Status.SUCCESS

        #         twist_msg.linear.x = self.direction[0]*self.max_linear_vel
        #         twist_msg.linear.y = self.direction[1]*self.max_linear_vel
        #         rospy.loginfo("[MOVE_DUMMY] update: RUNNING, moving for [%0.3f m]" %
        #                       remaining_dist)

        #         self.cmd_vel_pub.publish(twist_msg)
        #         return pt.common.Status.RUNNING
        #     rospy.loginfo(
        #         "[MOVE_DUMMY] update: SUCCESS, already moved")

        # elif self.avoid_collison:
        #     twist_msg.linear.x = self.blackboard.dir_to_avoid_collison[0] * \
        #         self.max_linear_vel
        #     twist_msg.linear.y = self.blackboard.dir_to_avoid_collison[1] * \
        #         self.max_linear_vel
        #     self.cmd_vel_pub.publish(twist_msg)
        #     rospy.loginfo("[MOVE_DUMMY] update: RUNNING, moving away from obstacle")
        #     return pt.common.Status.RUNNING
        # return pt.common.Status.SUCCESS

    def terminate(self, new_status):
        """
        terminate() is trigerred once the execution of the behavior finishes, 
        i.e. when the status changes from RUNNING to SUCCESS or FAILURE
        """
        rospy.loginfo("[MOVE_DUMMY] terminate: publishing zero linear velocity")
        twist_msg = Twist()
        twist_msg.linear.x = 0
        twist_msg.linear.y = 0
        self.cmd_vel_pub.publish(twist_msg)
        return super().terminate(new_status)

    def clbk_laser(self, data_laser):
        """
        Read sensor messagens, and determine distance to each region. 
        Manipulates the values measured by the sensor.
        Callback function for the subscription to the published Laser Scan values.
        """
        global run_env
        data= np.array(data_laser.ranges)
        if run_env==1:
            data = np.append(data,0.0)
        # laser scanner points which are outside the min and max range are assigned zero distance
        data[data == 0.0] = 20.0
        self.size = int(len(data))
        self.min_index = int(self.size*(self.direction+1)/4)
        self.max_index = int(self.size*(self.direction+3)/4)


        # Determine values for PD control of distance and P control of angle
        for i in range(self.min_index, self.max_index):
            if data[i] < data[self.min_index] and data[i] > 0.01:
                self.min_index = i
        self.angle_min = (self.min_index-self.size/2)*data_laser.angle_increment
        self.dist_min = data[self.min_index]
        self.dist_front = data[int(self.size/2)]
        self.diff_e = min((self.dist_min - self.wall_dist) - self.e, 100)
        self.e = min(self.dist_min - self.wall_dist, 100)
        # ONLY FOR SIMULATION
        # Determination of minimum distances in each region
        if run_env==0:
            self.regions = {
                'bright':   min(np.average(data[0:82]), self.inf),
                'right':    min(np.average(data[83:165]), self.inf),
                'fright':   min(np.average(data[166:248]), self.inf),
                'front':    min(np.average(data[249:331]), self.inf),
                'fleft':    min(np.average(data[332:414]), self.inf),
                'left':     min(np.average(data[415:497]), self.inf),
                'bleft':    min(np.average(data[498:580]), self.inf),
            }

        ##
        ##
        # ONLY FOR RUNNING ON ROBOT
        ##
        ##
        elif run_env==1:
            self.regions = {
                'bright':   min(np.average(data[0:63]), self.inf),
                'right':    min(np.average(data[63:125]), self.inf),
                'fright':   min(np.average(data[125:187]), self.inf),
                'front':    min(np.average(data[187:249]), self.inf),
                'fleft':    min(np.average(data[249:311]), self.inf),
                'left':     min(np.average(data[311:372]), self.inf),
                'bleft':    min(np.average(data[372:440]), self.inf),
            }


        #rospy.logself.info(regions)

        # Detection of Outer and Inner corner
        self.bool_outer_corner = self.is_outer_corner()
        self.bool_inner_corner = self.is_inner_corner()
        if self.bool_outer_corner == 0 and self.bool_inner_corner == 0:
            self.last_kinds_of_wall[self.index]=0

        # Indexing for last five pattern detection
        # This is latter used for low pass filtering of the patterns
        self.index = self.index + 1 #5 samples recorded to asses if we are at the corner or not
        if self.index == len(self.last_kinds_of_wall):
            self.index = 0

        self.take_action()

    def is_inner_corner(self):
        """
        Assessment of inner corner in the wall. 
        If the three front regions are self.inferior than the wall_dist.
        If all the elements in last_kinds_of_wall are 'I' and the last time a real corner was detected is superior or equal to 20 seconds:
            To state_outer_inner a 'I' is appended and 
            The time is restart.
        Returns:
                bool_inner_corner: 0 if it is not a inner corner; 1 if it is a inner corner
        """
        self.bool_inner_corner = 0
        if self.regions['fright'] < self.wall_dist and self.regions['front'] < self.wall_dist and self.regions['fleft'] < self.wall_dist:
            self.bool_inner_corner = 1
            self.last_kinds_of_wall[self.index]='I'
            self.elapsed_time = time.time() - self.last_inner_corner_detection_time # Elapsed time since last corner detection
            if self.last_kinds_of_wall.count('I') == len(self.last_kinds_of_wall) and self.elapsed_time >= 20:
                self.last_inner_corner_detection_time = time.time()
                self.loop_index_inner_corner = self.loop_index
                self.state_outer_inner = self.state_outer_inner[1:]
                self.state_outer_inner.append('I')
                print('It is a inner corner')
        return self.bool_inner_corner

    def is_outer_corner(self): 
        """
        Assessment of outer corner in the wall. 
        If all the regions except for one of the back regions are self.infinite then we are in the presence of a possible corner.
        If all the elements in last_kinds_of_wall are 'C' and the last time a real corner was detected is superior or equal to 30 seconds:
            To state_outer_inner a 'C' is appended and 
            The time is restart.
        Returns:
                bool_outer_corner: 0 if it is not a outer corner; 1 if it is a outer corner
        """
        self.bool_outer_corner = 0
        if (self.regions['fright'] == self.inf and self.regions['front'] == self.inf and self.regions['right'] == self.inf and self.regions['bright'] < self.inf  and self.regions['left'] == self.inf and self.regions['bleft'] == self.inf and self.regions['fleft'] == self.inf) or (self.regions['bleft'] < self.inf and self.regions['fleft'] == self.inf and self.regions['front'] == self.inf and self.regions['left'] == self.inf and self.regions['right'] == self.inf and self.regions['bright'] == self.inf and self.regions['fright'] == self.inf):
            self.bool_outer_corner = 1 # It is a corner
            self.last_kinds_of_wall[self.index]='C'
            self.elapsed_time = time.time() - self.last_outer_corner_detection_time # Elapsed time since last corner detection
            if self.last_kinds_of_wall.count('C') == len(self.last_kinds_of_wall) and self.elapsed_time >= 30:
                self.last_outer_corner_detection_time = time.time()
                self.loop_index_outer_corner = self.loop_index
                self.state_outer_inner = self.state_outer_inner[1:]
                self.state_outer_inner.append('C')
                print('It is a outer corner')
        return self.bool_outer_corner

    def change_direction(self):
        """
        Toggle direction in which the robot will follow the wall
            1 for wall on the left side of the robot and -1 for the right side
        """
        print('Change direction!')
        self.elapsed_time_dir = time.time() - self.last_change_direction_time # Elapsed time since last change direction
        if self.elapsed_time_dir >= 20:
            self.last_change_direction = time.time()
            self.direction = -self.direction # Wall in the other side now
            self.rotating = 1

    def rotating(self):
        """
        Rotation movement of the robot. 
        Returns:
                Twist(): msg with angular and linear velocities to be published
                        msg.linear.x -> 0m/s
                        msg.angular.z -> -2 or +2 rad/s
        """
        msg = Twist()
        msg.linear.x = 0
        msg.angular.z = self.direction*2
        return msg

    def random_wandering(self):
        """
        This function defines the linear.x and angular.z velocities for the random wandering of the robot.
        Returns:
                Twist(): msg with angular and linear velocities to be published
                        msg.linear.x -> [0.1, 0.3]
                        msg.angular.z -> [-1, 1]
        """
        msg = Twist()
        msg.linear.x = max(min( self.last_vel[0] + random.uniform(-0.01,0.01),0.3),0.1)
        msg.angular.z= max(min( self.last_vel[1] + random.uniform(-0.1,0.1),1),-1)
        if msg.angular.z == 1 or msg.angular.z == -1:
            msg.angular.z = 0
        self.last_vel[0] = msg.linear.x
        self.last_vel[1] = msg.angular.z
        return msg

    def following_wall(self):
        """
        PD control for the wall following state. 
        Returns:
                Twist(): msg with angular and linear velocities to be published
                        msg.linear.x -> 0; 0.5max_speed; 0.4max_speed
                        msg.angular.z -> PD controller response
        """
        msg = Twist()
        if self.dist_front < self.wall_dist:
            msg.linear.x = 0
        elif self.dist_front < self.wall_dist*2:
            msg.linear.x = 0.5*self.max_speed
        elif abs(self.angle_min) > 1.75:
            msg.linear.x = 0.4*self.max_speed
        else:
            msg.linear.x = self.max_speed
        if self.dist_min<self.wall_dist+0.1:
            msg.angular.z = max(min(self.direction*(self.p*self.e+self.d*self.diff_e) + self.angle*(self.angle_min-((math.pi)/2)*self.direction), 2.5), -2.5)
        elif self.dist_min>self.wall_dist+0.3:
            msg.angular.z = -0.2
        #print 'Turn Left angular z, linear x %f - %f' % (msg.angular.z, msg.linear.x)
        return msg
    
    def take_action(self):
        """
        Change state for the machine states in accordance with the active and inactive regions of the sensor.
                State 0 No wall found - all regions self.infinite - Random Wandering
                State 1 Wall found - Following Wall
                State 2 Pattern sequence reached - Rotating
        """

        msg = Twist()
        linear_x = 0
        angular_z = 0

        self.state_description = ''

        # Patterns for rotating
        rotate_sequence_V1 = ['I', 'C', 'C', 'C']
        rotate_sequence_V2 = [0, 'C', 'C', 'C']
        rotate_sequence_W  = ['I', 'C', 'I', 'C']

        if self.rotating == 1:
            self.state_description = 'case 2 - rotating'
            self.state = 2
            if(self.regions['left'] < self.wall_dist or self.regions['right'] < self.wall_dist):
                self.rotating = 0
        elif self.regions['fright'] == self.inf and self.regions['front'] == self.inf and self.regions['right'] == self.inf and self.regions['bright'] == self.inf and self.regions['fleft'] == self.inf and self.regions['left'] == self.inf and self.regions['bleft'] == self.inf:
            self.state_description = 'case 0 - random wandering'
            self.state = 0
        elif (self.loop_index == self.loop_index_outer_corner) and (rotate_sequence_V1 == self.state_outer_inner or rotate_sequence_V2 == self.state_outer_inner or rotate_sequence_W == self.state_outer_inner):
            self.state_description = 'case 2 - rotating'
            self.change_direction()
            self.state_outer_inner = [ 0, 0,  0, 'C']
            self.state = 2
        # Obstacle Avoidance
        elif self.regions['front'] < self.wall_dist or self.regions['fleft'] < self.wall_dist or self.regions['fright'] < self.wall_dist:
            self.state_description = 'case 3 - obstacle avoidance'
            self.state = 3
            if self.regions['fleft'] > self.wall_dist and self.regions['fright'] > self.wall_dist:
                self.state_description += ' - front clear'
                self.state = 1
                
        else:
            self.state_description = 'case 1 - following wall'
            self.state = 1
        self.blackboard.zState=self.state
    
    def obstacle_avoidance(self):
        # global state_description, regions_
        msg = Twist()
        msg.linear.x = 0  # move forward with a slower speed
        msg.angular.z = 0.3  # turn with a moderate speed
        self.state_description = 'case 3 - obstacle avoidance'
        # regions = regions_
        # ooo = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # print(regions)

        # check the regions for obstacles
        if self.regions['front'] < self.wall_dist and self.regions['fleft'] < self.wall_dist and self.regions['fright'] < self.wall_dist:
            # if the front and both sides have obstacles, turn around
            msg.linear.x = 0.1
            msg.angular.z = 0.3  # turn with the opposite direction
            print("front fleft fright")
        elif self.regions['front'] < self.wall_dist and self.regions['fleft'] < self.wall_dist:
            # if the front and left side have obstacles, turn right
            msg.linear.x = 0.1
            msg.angular.z =  0.3  # turn right
            print("front fleft")
        elif self.regions['front'] < self.wall_dist and self.regions['fright'] < self.wall_dist:
            # if the front and right side have obstacles, turn left
            msg.linear.x = 0.1
            msg.angular.z = 0.3  # turn left
            print("front fright")
        elif self.regions['fright'] < self.wall_dist:
            # if only the right side has obstacles, turn left
            msg.linear.x = 0.1
            msg.angular.z = 0.3  # turn left
            print("  fright")
        elif self.regions['fleft'] < self.wall_dist:
            # if only the left side has obstacles, turn right
            msg.linear.x = 0.1
            msg.angular.z = 0.3  # turn right
            print(" fleft ")
        
        self.pub.publish(msg)