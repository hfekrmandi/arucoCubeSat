import numpy as np
import cv2
import cv2.aruco as aruco


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)
    # 480x640 is the web camera resolution

    # The two lines can be interchanged to choose if the image is in grayscale or not. Grayscale saves RAM, but color makes the drawn markers easier to see.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = frame
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()


    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(corners)

    frame = aruco.drawDetectedMarkers(gray, corners, ids, (255,255,255))

    # print(rejectedImgPoints)
    cv2.imshow('Camera View (Press \'q\' to quit)',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
