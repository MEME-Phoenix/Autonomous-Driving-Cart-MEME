import time
from Motor import *
import rospy
from Ultrasonic import *

PWM = Motor()
ultrasonic = Ultrasonic()

x, y, size = 0, 0, 0
wait_time = 0.5

def callback(msg):
    global x, y, size
    x = msg.linear.x
    y = msg.linear.y
    size = msg.linear.size

def forward():
    PWM.setMotorModel(1000, 1000, 1000, 1000)  # Forward
    print("The car is moving forward")
    time.sleep(wait_time)


def backward():
    PWM.setMotorModel(-1000, -1000, -1000, -1000)  # Back
    print("The car is going backwards")
    time.sleep(wait_time)


def left():
    PWM.setMotorModel(-1500, -1500, 2000, 2000)  # Left
    print("The car is turning left")
    time.sleep(wait_time)


def right():
    PWM.setMotorModel(2000, 2000, -1500, -1500)  # Right
    print("The car is turning right")
    time.sleep(wait_time)


def stop():
    PWM.setMotorModel(0, 0, 0, 0)  # Stop


def alphabot_nav(x, y):
    d = ultrasonic.get_distance()
    print("distance: %s" % d)
    if size > 40:
        if x > 65:
            print("move right")
            right()
            time.sleep(0.1)
            stop()
        elif x < 35:
            print("move left")
            left()
            time.sleep(0.1)
            stop()
        else:
            print("move forward")
            forward()
            time.sleep(0.1)
            stop()
    else:
        print("stop")
        stop()


def main():
    rospy.init_node('alphabot_track')
    sub = rospy.Subscriber('/deepsort_result', DeepSortResult, callback)

    while (True):
        alphabot_nav(x, y)


if __name__ == '__main__':
    main()

