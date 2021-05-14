

"""
Course:  Training YOLO v3 for Objects Detection with Custom Data

Section-1
Quick Win - Step 1: Simple Object Detection by thresholding with mask
File: finding-mask.py
"""


# Convenient way for choosing right Colour Mask to Detect needed Object
#
# Algorithm:
# Reading RGB image --> Converting to HSV --> Getting Mask
#
# Result:
# min_blue, min_green, min_red = 21, 222, 70
# max_blue, max_green, max_red = 176, 255, 255


# Importing needed library
import cv2


# Preparing Track Bars
# Defining empty function
def do_nothing(x):
    pass


# Giving name to the window with Track Bars
# And specifying that window is resizable
cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

# Defining Track Bars for convenient process of choosing colours
# For minimum range
cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('min_red', 'Track Bars', 0, 255, do_nothing)

# For maximum range
cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_green', 'Track Bars', 0, 255, do_nothing)
cv2.createTrackbar('max_red', 'Track Bars', 0, 255, do_nothing)

# while True:
#     if cv2.waitKey(0):
#         break

# Reading image with OpenCV library
# In this way image is opened already as numpy array
# WARNING! OpenCV by default reads images in BGR format
image_BGR = cv2.imread('2021-03-26-200546.jpg')
# Resizing image in order to use smaller windows
image_BGR = cv2.resize(image_BGR, (600, 426))

# Showing Original Image
# Giving name to the window with Original Image
# And specifying that window is resizable
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image_BGR)

# Converting Original Image to HSV
image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

# Showing HSV Image
# Giving name to the window with HSV Image
# And specifying that window is resizable
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)
cv2.imshow('HSV Image', image_HSV)


# Defining loop for choosing right Colours for the Mask
while True:
    # Defining variables for saving values of the Track Bars
    # For minimum range
    min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
    min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
    min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

    # For maximum range
    max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
    max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
    max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

    # Implementing Mask with chosen colours from Track Bars to HSV Image
    # Defining lower bounds and upper bounds for thresholding
    mask = cv2.inRange(image_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    # Showing Binary Image with implemented Mask
    # Giving name to the window with Mask
    # And specifying that window is resizable
    cv2.namedWindow('Binary Image with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image with Mask', mask)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Destroying all opened windows
cv2.destroyAllWindows()


# Printing final chosen Mask numbers
print('min_blue, min_green, min_red = {0}, {1}, {2}'.format(min_blue, min_green,
                                                            min_red))
print('max_blue, max_green, max_red = {0}, {1}, {2}'.format(max_blue, max_green,
                                                            max_red))


"""
Some comments

HSV (hue, saturation, value) colour-space is a model
that is very useful in segmenting objects based on the colour.

With OpenCV function cv2.inRange() we perform basic thresholding operation
to detect an object based on the range of pixel values in the HSV colour-space.
"""
