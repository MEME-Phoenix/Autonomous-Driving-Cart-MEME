
# Detecting Object with chosen Colour Mask
#
# Algorithm:
# Reading RGB image --> Converting to HSV --> Implementing Mask -->
# --> Finding Contour Points --> Extracting Rectangle Coordinates -->
# --> Drawing Bounding Box --> Putting Label
#
# Result:
# Window with Detected Object, Bounding Box and Label in Real Time


# Importing needed library
import cv2

# Defining lower bounds and upper bounds of founded Mask
min_blue, min_green, min_red = 40, 70, 180
max_blue, max_green, max_red = 70, 100, 240

# Getting version of OpenCV that is currently used
# Converting string into the list by dot as separator
# and getting first number
v = cv2.__version__.split('.')[0]

# Defining object for reading video from camera
camera = cv2.VideoCapture(0)

# Defining loop for catching frames
while True:
    # Capture frame-by-frame from camera
    _, frame_BGR = camera.read()

    # Converting current frame to HSV
    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # Implementing Mask with founded colours from Track Bars to HSV Image
    mask = cv2.inRange(frame_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    # Showing current frame with implemented Mask
    # Giving name to the window with Mask
    # And specifying that window is resizable
    cv2.namedWindow('Binary frame with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary frame with Mask', mask)

    # Finding Contours
    # Pay attention!
    # Different versions of OpenCV returns different number of parameters
    # when using function cv2.findContours()

    # In OpenCV version 3 function cv2.findContours() returns three parameters:
    # modified image, found Contours and hierarchy
    # All found Contours from current frame are stored in the list
    # Each individual Contour is a Numpy array of(x, y) coordinates
    # of the boundary points of the Object
    # We are interested only in Contours

    # Checking if OpenCV version 3 is used
    if v == '3':
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # In OpenCV version 4 function cv2.findContours() returns two parameters:
    # found Contours and hierarchy
    # All found Contours from current frame are stored in the list
    # Each individual Contour is a Numpy array of(x, y) coordinates
    # of the boundary points of the Object
    # We are interested only in Contours

    # Checking if OpenCV version 4 is used
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Finding the biggest Contour by sorting from biggest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extracting Coordinates of the biggest Contour if any was found
    if contours:
        # Getting rectangle coordinates and spatial size from biggest Contour
        # Function cv2.boundingRect() is used to get an approximate rectangle
        # around the region of interest in the binary image after Contour was found
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Drawing Bounding Box on the current BGR frame
        cv2.rectangle(frame_BGR, (x_min - 15, y_min - 15),
                      (x_min + box_width + 15, y_min + box_height + 15),
                      (0, 255, 0), 3)

        # Preparing text for the Label
        label = 'Detected Object'

        # Putting text with Label on the current BGR frame
        cv2.putText(frame_BGR, label, (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Showing current BGR frame with Detected Object
    # Giving name to the window with Detected Object
    # And specifying that window is resizable
    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', frame_BGR)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Destroying all opened windows
cv2.destroyAllWindows()


"""
Some comments

With OpenCV function cv2.findContours() we find 
contours of white object from black background.

There are three arguments in cv.findContours() function,
first one is source image, second is contour retrieval mode,
third is contour approximation method.


In OpenCV version 3 three parameters are returned:
modified image, the contours and hierarchy.
Further reading about Contours in OpenCV v3:
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html


In OpenCV version 4 two parameters are returned:
the contours and hierarchy.
Further reading about Contours in OpenCV v4:
https://docs.opencv.org/4.0.0/d4/d73/tutorial_py_contours_begin.html


Contours is a Python list of all the contours in the image.
Each individual contour is a Numpy array of (x,y) coordinates 
of boundary points of the object.

Contours can be explained simply as a curve joining all the 
continuous points (along the boundary), having same colour or intensity.
"""
