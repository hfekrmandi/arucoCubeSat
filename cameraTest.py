# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time, cv2
import numpy as np
import cv2.aruco as aruco
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1280, 960)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(1280, 960))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = frame.array

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, dst=None, flipCode=-1) 
    #gray = frame
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()


    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''

    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    print(corners)

    frame = aruco.drawDetectedMarkers(frame, corners, ids, (255,255,255))
    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
