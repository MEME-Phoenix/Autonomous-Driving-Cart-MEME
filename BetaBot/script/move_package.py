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
    elif pos == "right":
        for i in range(4):
            led.ledIndex(p[i+4],color[0], color[1], color[2])
    # time.sleep(sleep_time)


PWM=Motor()
ultrasonic=Ultrasonic()    

sleep_time=0.3
def forward():
    PWM.setMotorModel(1000, 1000, 1000, 1000)  # Forward
    set_led("all", [0,255,0])
    print("The car is moving forward")
    time.sleep(sleep_time)
def backward():
    PWM.setMotorModel(-1000, -1000, -1000, -1000)  # Back
    print("The car is going backwards")
    time.sleep(sleep_time)
def left():
    PWM.setMotorModel(-1500, -1500, 2000, 2000)  # Left
    print("The car is turning left")
    set_led("left", [0,255,0])
    time.sleep(sleep_time)
def right():
    PWM.setMotorModel(2000, 2000, -1500, -1500)  # Right
    set_led("right", [0,255,0])
    print("The car is turning right")
    time.sleep(sleep_time)
def stop():
    PWM.setMotorModel(0, 0, 0, 0)  # Stop
    set_led("all", [255,0,0])
    time.sleep(sleep_time)

def alphabot_nav(x, y):
    d = ultrasonic.get_distance()
    print("distance: %s"%d)
    if d >30:
        if x>65:
            print("move right")
            right()
            time.sleep(0.1)
            stop()
        elif x<35:
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

