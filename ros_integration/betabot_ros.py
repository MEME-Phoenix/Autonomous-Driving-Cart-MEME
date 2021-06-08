#!/usr/bin/env python

import tf
import rospy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import time, math

status_code = 0
front_ultra = 0

def callback(msg):
    # follows the conventional x, y, poses
    global status_code, front_ultra
    status_code = msg.pose.pose.position.x
    front_ultra = msg.pose.pose.position.y
    print('status code:',str(status_code), '\tfront ultrasonic distance:', front_ultra)

def publish_status():

    publisher = rospy.Publisher('betabot_status', Odometry, queue_size=1)
    msg = Twist()
    msg.pose.pose.position.x = status_code
    msg.pose.pose.position.y = front_ultra
    publisher.publish(msg)

def betabot_result_to_ros(x, y):
    rospy.init_node('betabot_status')
    odom_sub = rospy.Subscriber('/deepsort_result', Odometry, callback)

# if __name__ == "__main__":
#     rospy.init_node('deepsort_result')
#     odom_sub = rospy.Subscriber('/betabot_status', Odometry, callback)
    # publish_cmd_vel(x, y)
