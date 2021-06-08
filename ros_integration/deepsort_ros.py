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
    print('status code:',str(status_code), 'front ultrasonic distance:', front_ultra)

def publish_cmd_vel(x, y):
    goal = 0.6
    speed = 0.01
    publisher = rospy.Publisher('result_cmd_vel', Twist, queue_size=1)
    cmd = Twist()
    cmd.linear.x = x
    cmd.linear.y = y
    cmd.linear.z = 0
    rospy.sleep(1)
    seconds = time.time()
    if front_ultra < 10:
        print('Too close to object in front')
    else:
        while time.time() - seconds < goal / speed:
            publisher.publish(cmd)
def deepsort_result_to_ros(x, y):
    rospy.init_node('deepsort_status')
    odom_sub = rospy.Subscriber('/betabot_status', Odometry, callback)

# if __name__ == "__main__":
#     rospy.init_node('deepsort_result')
#     odom_sub = rospy.Subscriber('/betabot_status', Odometry, callback)
    # publish_cmd_vel(x, y)
