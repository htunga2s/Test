#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

def talker():
    '''
    This node can be used to initiate different publishers when running the scripts on local machine.
    '''
    pub1 = rospy.Publisher('/mileage', Float32, queue_size=1)
    pub2 = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
