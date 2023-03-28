import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def scancb(msg):
    ranges = np.array(msg.ranges)
    len_of_scan = len(ranges)

    midle_point_idx = int(len_of_scan/2)
    midle_value = ranges[midle_point_idx]
    minimum_value = min(ranges)
    # print(ranges)
    # print(minimum_value)

    # while minimum_value < ranges[midle_point_idx + 10]:
    for i in np.arange(ranges[midle_point_idx - 10],ranges[midle_point_idx + 10]):
        # print("high")
        if minimum_value == i:
            print("break")
            return 0
        rotate()
    cmd_pub.publish(Twist())

def rotate():
    velocity = Twist()
    velocity.linear.x = 0.0
    velocity.angular.z = 0.25
    cmd_pub.publish(velocity)

if __name__ == '__main__':
    rospy.init_node('wall_following', anonymous=True)
    scan_sub = rospy.Subscriber('/scan_filtered', LaserScan, scancb)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.spin()
    # try:
    #     main()
    # except rospy.exceptions. ROSInitException:
    #     print('Terminate')