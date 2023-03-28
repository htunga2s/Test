#!/usr/bin/env python

import functools
import py_trees as pt
import py_trees_ros as ptr
import py_trees.console as console
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import rospy
import sys
import operator
from behaviors import *


def create_root():
    """
    Method to construct the behavior tree
    """
    # initialising all required behaviors
    root = pt.composites.Parallel("root")

    topics2bb = pt.composites.Sequence("Topics2BB")
    priorities = pt.composites.Selector("Priorities")
    navigation_stack = pt.composites.Sequence("NavigationStack")    

    battery2bb = batteryStatus2bb(name="Battery2BB",
                                   topic_name="/mileage",
                                   threshold=30.0)
    laser_scan2bb = laserScanFiltered2bb(name="LaserScan2BB",
                                           topic_name="/scan_filtered",
                                           safe_range=0.2
                                           )
    odom_data2bb = odom2bb(name="OdomData2BB",
                           topic_name="/odom")

    mapping = pt.composites.Sequence("Map")
    localisation = pt.composites.Sequence("Localisation")
    navigation = pt.composites.Sequence("Navigate")

    gmapping = launchNodes(name="GMapping", launch_file="gmapping.launch", mapping_time_out=150)
    save_map = saveMap(name="SaveMap", map_name="sim_env")
    amcl = launchNodes(name="AMCL", launch_file="amcl.launch")
    move_base_dwa = launchNodes(name="MoveBaseDWA", launch_file="move_base_dwa.launch")

   
    wait_for_battery = pt.meta.success_is_failure(ptr.subscribers.WaitForData)(name="WaitForBattery",
                                                                               topic_name='/mileage',
                                                                               topic_type=Float32,
                                                                               clearing_policy=pt.common.ClearingPolicy.NEVER)
    wait_for_laser_scan = pt.meta.success_is_failure(ptr.subscribers.WaitForData)(name="WaitForLaserScan",
                                                                                  topic_name='/scan_filtered',
                                                                                  topic_type=LaserScan,
                                                                                  clearing_policy=pt.common.ClearingPolicy.NEVER)
    wait_for_odom = pt.meta.success_is_failure(ptr.subscribers.WaitForData)(name="WaitForOdom",
                                                                            topic_name='/odom',
                                                                            topic_type=Odometry,
                                                                            clearing_policy=pt.common.ClearingPolicy.NEVER)

    collison_emergency = pt.meta.success_is_failure(
        pt.composites.Selector)(name="CollisonEmergency")
    battery_emergency = pt.meta.success_is_failure(
        pt.composites.Selector)(name="BatteryEmergency")
    move_align_follow = pt.meta.success_is_failure(
        pt.composites.Sequence)(name="MoveAlignFollow")

    avoid_collison = move(name="AvoidCollison", topic_name="/cmd_vel",
                          avoid_collison=True, max_linear_vel=0.2)
    is_collison_possible = pt.blackboard.CheckBlackboardVariable(
        name="Possible Collison?",
        variable_name='collison_warning',
        expected_value=False
    )

    is_battery_ok = pt.blackboard.CheckBlackboardVariable(
        name="BatteryOk?",
        variable_name="battery_low_warning",  # updated in the pt.battery.ToBlackboard
        expected_value=False  # expecting battery_low_warning to be False
    )
    rotate_platform = rotate(name="RotatePlatform",
                             topic_name="/cmd_vel", max_ang_vel=0.2)

    time_up=pt.blackboard.CheckBlackboardVariable(
        name='TimeUp?',
        variable_name='timeup',
        expected_value=False
                )
    #RANSAC
    ransac = wallParamRANSAC2bb(
        name="Ransac", topic_name="/scan_filtered", wall_priority_thresh=0.8)
    
    #move towards WALL
    align_according_to_wall = rotate_dummy(name="AlignToWall", topic_name="/cmd_vel",
                           to_align=True, max_ang_vel=0.5, align_threshold=0.2) # main_wall_facer()
    #move along wall
    move_along_wall = move_dummy(name="MoveAlongWall",
                           topic_name="/cmd_vel", along_wall=True, max_linear_vel=0.5)

    is_wall_param_available = pt.blackboard.WaitForBlackboardVariable(
        name="WallParamAvailable?",
        variable_name="num_walls",
        expected_value=0,  # number of walls detected should be greater than zero
        comparison_operator=operator.gt
    )

    idle = pt.behaviours.Running(name="Idle")

    #----------------------------------------------------------------------------------------
    # Tree formation
    #----------------------------------------------------------------------------------------
    root.add_children([topics2bb, navigation_stack, priorities])

    topics2bb.add_children([laser_scan2bb, odom_data2bb, battery2bb])
    navigation_stack.add_children([mapping, localisation, navigation])
    priorities.add_children(
        [wait_for_battery, wait_for_laser_scan, wait_for_odom,collison_emergency, battery_emergency, move_align_follow, idle])

    mapping.add_children([gmapping, save_map])
    localisation.add_child(amcl)
    navigation.add_children([move_base_dwa])
    collison_emergency.add_children(
        [is_collison_possible, avoid_collison])
        
    battery_emergency.add_children([is_battery_ok, rotate_platform])
    move_align_follow.add_children(
        [time_up,ransac,is_wall_param_available, align_according_to_wall, move_along_wall])
    return root


def shutdown(behaviour_tree):
    """
    This method will be called on termination of the behavior tree
    """
    rospy.loginfo("[SHUTDOWN] shutting down the behavior_tree")
    behaviour_tree.interrupt()


def main():
    """
    Main function initiates behavior tree construction
    """
    # Initialising the node with name "behavior_tree"
    try:
        rospy.init_node("behavior_tree")
        root = create_root()
        behaviour_tree = ptr.trees.BehaviourTree(root)
        rospy.on_shutdown(functools.partial(shutdown, behaviour_tree))
        if not behaviour_tree.setup(timeout=15):
            console.logerror("failed to setup the tree, aborting.")
            sys.exit(1)

        def tick_printer(t): return pt.display.print_ascii_tree(
            t.root, show_status=True)

        behaviour_tree.tick_tock(sleep_ms=50, post_tick_handler=tick_printer)

    except KeyboardInterrupt:
        rospy.signal_shutdown(
            "Shutting down node due to keyboard interruption")


if __name__ == '__main__':
    main()

