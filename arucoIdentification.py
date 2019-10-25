from picamera.array import PiRGBArray
from picamera import PiCamera
import time, cv2
import numpy as np
import cv2.aruco as aruco

shape = (1280, 960)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = shape
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=shape)
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = frame.array

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.flip(frame, dst=None, flipCode=-1) 
    

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''

    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    print(corners)

    frame = aruco.drawDetectedMarkers(frame, corners, ids, (255,255,255))
    
    # show the frame
    cv2.imshow('Camera View (Press \'q\' to quit)', frame)
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if (key == ord("q")):
        break

cv2.destroyAllWindows()
