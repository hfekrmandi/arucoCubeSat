import numpy as np
import cv2
import cv2.aruco as aruco
import pickle

# side length of tag in meters
markerLength = 0.09 #0.2032

# import saved calibration information
cal = pickle.load( open( "calibrationSave2.p", "rb" ) )
retval, cameraMatrix, distCoeffs, rvecsUnused, tvecsUnused = cal

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 12.0, (640,480), False)
font = cv2.FONT_HERSHEY_SIMPLEX

def drawCrosshairs(frame):
    y, x = frame.shape
    cv2.line(frame,(int(x/2)+10, int(y/2)),(int(x/2)-10, int(y/2)),(255,255,255),2)
    cv2.line(frame,(int(x/2), int(y/2)+10),(int(x/2), int(y/2)-10),(255,255,255),2)

crosshairs = True
recording = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)
    # 480x640 is the web camera resolution

    # Uncomment to convert to grayscale, which saves RAM but makes drawn markers more difficult to see
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (crosshairs):
        drawCrosshairs(frame)
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()


    # detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints]]]]) -> corners, ids, rejectedImgPoints
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    frame = aruco.drawDetectedMarkers(frame, corners, ids, (255,255,255))

    
    # estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]]) -> rvecs, tvecs, _objPoints
    rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, None, None, None)

    
    # drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length) -> image
    
    if (ids is not None):
        print(tvecs)
        for i in range(len(rvecs)):
            frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength / 2)

    if (recording):
        out.write(frame)
        cv2.putText(frame,'RECORDING',(0,30), font, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('Camera View (\'q\' to quit, \'c\' to toggle crosshairs, \'r\' to record)', frame)

    char = cv2.waitKey(1) & 0xFF
    if (char == ord('q')):
        break
    elif (char == ord('c')):
        crosshairs = not crosshairs
    elif (char == ord('r')):
        recording = not recording
        

cap.release()
out.release()
cv2.destroyAllWindows()
