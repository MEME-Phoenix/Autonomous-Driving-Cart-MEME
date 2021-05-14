#!/usr/bin/env python

from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2
import time

x_pos = 0
y_pos = 0

def read_rgb_image(image_name, show):
    rgb_image = cv2.imread(image_name)
    if show: 
        cv2.imshow("RGB Image",rgb_image)
    return rgb_image

def filter_color(rgb_image, lower_bound_color, upper_bound_color):
    #convert the image into the HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv image",hsv_image)

    #define a mask using the lower and upper bounds of the yellow color 
    mask = cv2.inRange(hsv_image, lower_bound_color, upper_bound_color)

    return mask


    

def getContours(binary_image):     
    #_, contours, hierarchy = cv2.findContours(binary_image, 
    #                                          cv2.RETR_TREE, 
    #                                           cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(binary_image.copy(), 
                                            cv2.RETR_EXTERNAL,
	                                        cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_ball_contour(binary_image, rgb_image, contours):
    global x_pos, y_pos
    black_image = np.zeros([binary_image.shape[0], binary_image.shape[1],3],'uint8')
    
    for c in contours:
        area = cv2.contourArea(c)
        print("area : ", str(area))
        perimeter= cv2.arcLength(c, True)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if (area>3000):
            cv2.drawContours(rgb_image, [c], -1, (150,250,150), 1)
            cv2.drawContours(black_image, [c], -1, (150,250,150), 1)
            cx, cy = get_contour_center(c)
            print("cx : ", str(cx/640), "cy : ", str(cy/480) )
            x_pos = cx/640*100
            y_pos = cy/480*100
            cv2.circle(rgb_image, (cx,cy),(int)(radius),(0,0,255),1)
            cv2.circle(black_image, (cx,cy),(int)(radius),(0,0,255),1)
            cv2.circle(black_image, (cx,cy),5,(150,150,255),-1)
            #print ("Area: {}, Perimeter: {}".format(area, perimeter))
        else:
            x_pos = -1
            y_pos = -1

    #print ("number of contours: {}".format(len(contours)))
    cv2.imshow("RGB Image Contours",rgb_image)
    cv2.imshow("Black Image Contours",black_image)

def get_contour_center(contour):
    M = cv2.moments(contour)
    cx=-1
    cy=-1
    if (M['m00']!=0):
        cx= int(M['m10']/M['m00'])
        cy= int(M['m01']/M['m00'])
    return cx, cy

def detect_ball_in_a_frame(image_frame):
    yellowLower =(0,178,103)
    yellowUpper = (190,255,246)
    rgb_image = image_frame
    binary_image_mask = filter_color(rgb_image, yellowLower, yellowUpper)
    contours = getContours(binary_image_mask)
    draw_ball_contour(binary_image_mask, rgb_image,contours)





import time
from Motor import *            
from Ultrasonic import *

import time
from Led import *
led=Led()

def set_led(pos, color):
    p = [0x20, 0x10,0x08, 0x04, 0x01, 0x02, 0x40, 0x80]
    if pos =="all":
        for i in p:
            led.ledIndex(i,color[0], color[1], color[2])
    elif pos == "left":
        for i in range(4):
            led.ledIndex(p[i],color[0], color[1], color[2])
        for i in range(4):
            led.ledIndex(p[i+4],0, 0, 0)
    elif pos == "right":
        for i in range(4):
            led.ledIndex(p[i],0, 0, 0)
        for i in range(4):
            led.ledIndex(p[i+4],color[0], color[1], color[2])
    # time.sleep(sleep_time)


PWM=Motor()
ultrasonic=Ultrasonic()    

sleep_time=1/32
disatance_tolerance = 15
motor_speed = 0
def forward():
    PWM.setMotorModel(500, 500, 500, 500)  # Forward
    set_led("all", [0,20,0])
    print("The car is moving forward")
    time.sleep(sleep_time)
def backward():
    PWM.setMotorModel(-500, -500, -500, -500)  # Back
    print("The car is going backwards")
    time.sleep(sleep_time)
def left():
    PWM.setMotorModel(-750, -750, 1000, 1000)  # Left
    print("The car is turning left")
    set_led("left", [0,10,0])
    time.sleep(sleep_time)
def right():
    PWM.setMotorModel(1000, 1000, -750, -750)  # Right
    set_led("right", [0,10,0])
    print("The car is turning right")
    time.sleep(sleep_time)
def stop_distance():
    PWM.setMotorModel(0, 0, 0, 0)  # Stop
    set_led("all", [10,0,0])
    print("stop due to distance")
def stop_noone():
    PWM.setMotorModel(0, 0, 0, 0)  # Stop
    set_led("all", [10,10,0])
    print("stop du to no one detected")


def alphabot_nav(x, y):
    d = ultrasonic.get_distance()
    print("distance: %s"%d)
    if d< disatance_tolerance:
        stop_distance()
    
    elif d >disatance_tolerance and x>=0 and y>=0:
        if x>65:
            print("move right")
            right()
            # stop()
        elif x<35:
            print("move left")
            left()
            # stop()
        else:
            print("move forward")
            forward()
            # stop()
    
    else:
        stop_noone()



def main():

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)


    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text

        detect_ball_in_a_frame(frame.array)
        time.sleep(sleep_time)
        alphabot_nav(x_pos, y_pos)

        # image = frame.array
        # # show the frame
        # cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



cv2.waitKey(0)
cv2.destroyAllWindows()