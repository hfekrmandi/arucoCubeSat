import numpy as np
import cv2
import cv2.aruco as aruco

'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''

# get input from user, verify value is valid
validValue = False
while(validValue != True):
    try:
        value=int(input('Please input integer from 0 to 249 for tag value: '))
    except ValueError:
        print("Not a number")
    else:
        if(value >= 0):
            validValue = True
        else:
            print("Number must be greater or equal to zero")

'''
More tag versions are available and listed below.

DICT_4X4_100
DICT_4X4_1000
DICT_4X4_250
DICT_4X4_50
DICT_5X5_100
DICT_5X5_1000
DICT_5X5_250
DICT_5X5_50
DICT_6X6_100
DICT_6X6_1000
DICT_6X6_250
DICT_6X6_50
DICT_7X7_100
DICT_7X7_1000
DICT_7X7_250
DICT_7X7_50
'''


# tag restricted to 6x6 blocks plus 1 block thick black perimeter
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# second parameter is id number, last parameter is total image size
img = aruco.drawMarker(aruco_dict, value, 640)
cv2.imwrite("aruco_marker_" + str(value) +".jpg", img)
 
cv2.imshow('Generated Aruco Marker (Press any key to quit)',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
