import numpy as np
import cv2
from cv2 import aruco
import pickle
import datetime
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import my_socket
import socket

# side length of tag in meters
markerLength = 0.1
# import saved calibration information
# calibrationSave.p should be correct for laptop webcam
cal = pickle.load(open("calibrationSave.p", "rb"))
retval, cameraMatrix, distCoeffs, rvecsUnused, tvecsUnused = cal
para = aruco.DetectorParameters_create()
para.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 12.0, (640,480), False)
font = cv2.FONT_HERSHEY_SIMPLEX

x_permanent = [0]
y_permanent = [0]
z_permanent = [0]
t1_permanent = [0]
t2_permanent = [0]
t3_permanent = [0]
prev_time = datetime.datetime.now()
times = [0]
n_stored = 0
n_stored_max = 10
n_points_plotting = 100


def plot_real_time():
    plt.clf()
    plt.subplot(211)
    plt.plot(times, t1_permanent, 'r-', lw=2, label='Theta 1') # Pitch
    plt.plot(times, t2_permanent, 'b-', lw=2, label='Theta 2') # Roll
    plt.plot(times, t3_permanent, 'k-', lw=2, label='Theta 3') # Yaw
    plt.xlabel('time')
    plt.ylabel('Angle (degrees)')
    plt.title('Angles')
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(times, x_permanent, 'r-', lw=2, label='x')
    plt.plot(times, y_permanent, 'b-', lw=2, label='y')
    plt.plot(times, z_permanent, 'k-', lw=2, label='z')
    plt.xlabel('time')
    plt.ylabel('Distance (cm)')
    plt.title('Distance')
    plt.legend()
    plt.grid(True)

    plt.show(block=False)
    plt.pause(0.01)


def plot_real_time_3d(X, Y, Z, R, ax):
    ax.cla()
    xy_range = 0.1
    z_range = 1
    length = 0.05

    for i in range(len(X)):
        #R_X = np.array([[1, 0, 0], [0, np.cos(p[i]), -np.sin(p[i])], [0, np.sin(p[i]), np.cos(p[i])]])
        #R_Y = np.array([[np.cos(r[i]), 0, np.sin(r[i])], [0, 1, 0], [-np.sin(r[i]), 0, np.cos(r[i])]])
        #R_Z = np.array([[np.cos(y[i]), -np.sin(y[i]), 0], [np.sin(y[i]), np.cos(y[i]), 0], [0, 0, 1]])

        [U, V, W] = (length * R).T

        colors = ['b', 'g', 'r']
        for j in range(3):
            ax.quiver(X[i], Y[i], Z[i], U[j], V[j], W[j], color=colors[j])

    ax.set_xlim([-xy_range, xy_range])
    ax.set_ylim([-xy_range, xy_range])
    ax.set_zlim([0, z_range])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show(block=False)
    plt.pause(0.01)


def drawCrosshairs(frame):
    y, x = frame.shape
    cv2.line(frame,(int(x/2)+10, int(y/2)),(int(x/2)-10, int(y/2)),(255,255,255),2)
    cv2.line(frame,(int(x/2), int(y/2)+10),(int(x/2), int(y/2)-10),(255,255,255),2)


# Turn features on/off
crosshairs = True
recording = False
plot = True
plot3d = False
send_data = False
if send_data:
    matlab_socket = my_socket.MySocket()
    matlab_socket.connect('127.0.0.1', 80)

if plot3d:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)
    # 480x640 is the web camera resolution

    # Uncomment to convert to grayscale, which saves RAM but makes drawn markers more difficult to see
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (crosshairs):
        drawCrosshairs(frame)

    # This is where you set what type pf tag to use: aruco.DICT_NXN_250
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()


    # detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints]]]]) -> corners, ids, rejectedImgPoints
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    frame = aruco.drawDetectedMarkers(frame, corners, ids, (255, 255, 255))

    
    # estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]]) -> rvecs, tvecs, _objPoints
    rvecs, tvecs, objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, None, None, None)

    
    # drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length) -> image
    
    if (ids is not None):
        print('translation vectors')
        print(tvecs)
        print('rotation vectors')
        #print(rvecs)
        #print()
        for i in range(len(rvecs)):
            frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength / 2)

        for rot in rvecs:
            rvec_rodr = np.eye(3)
            rvec_rodr = cv2.Rodrigues(rot, rvec_rodr)
            #print(rvec_rodr[0])
            angles = np.zeros([0, 2])
            theta = -np.arcsin(rvec_rodr[0][2][0])
            psi = np.arctan2(rvec_rodr[0][2][1] / np.cos(theta), rvec_rodr[0][2][2] / np.cos(theta))
            phi = np.arctan2(rvec_rodr[0][1][0] / np.cos(theta), rvec_rodr[0][0][0] / np.cos(theta))
            psi *= 180 / np.pi
            theta *= 180 / np.pi
            phi *= 180 / np.pi
            print([psi, theta, phi])
            ## [psi, theta, phi] rotate about
            ## [  y,     x,   z] respectively

        print()


        if plot:
            cur_time = datetime.datetime.now()
            delta = cur_time - prev_time
            if n_stored < n_stored_max:
                n_stored += 1
                delta_secs = delta.seconds + 0.000001 * delta.microseconds
                [x, y, z] = tvecs[0][0]
                x_permanent.append(x)
                y_permanent.append(y)
                z_permanent.append(z)
                if psi < 0:
                    t1_permanent.append(psi + 180)
                else:
                    t1_permanent.append(psi - 180)
                t2_permanent.append(theta)
                t3_permanent.append(phi)
                times.append(delta_secs)

                x_permanent = x_permanent[-n_points_plotting:]
                y_permanent = y_permanent[-n_points_plotting:]
                z_permanent = z_permanent[-n_points_plotting:]
                t1_permanent = t1_permanent[-n_points_plotting:]
                t2_permanent = t2_permanent[-n_points_plotting:]
                t3_permanent = t3_permanent[-n_points_plotting:]
                times = times[-n_points_plotting:]
            else:
                n_stored = 0
                plot_real_time()

        if plot3d:
            tvecs_shape = np.shape(tvecs)
            x = np.zeros([tvecs_shape[0]])
            y = np.zeros([tvecs_shape[0]])
            z = np.zeros([tvecs_shape[0]])
            for i in range(tvecs_shape[0]):
                [x[i], y[i], z[i]] = tvecs[i][0]
            plot_real_time_3d(x, y, z, rvec_rodr[0], ax)

        if send_data:
            tvecs_shape = np.shape(tvecs)
            x = np.zeros([tvecs_shape[0]])
            y = np.zeros([tvecs_shape[0]])
            z = np.zeros([tvecs_shape[0]])
            t1 = np.zeros([tvecs_shape[0]])
            t2 = np.zeros([tvecs_shape[0]])
            t3 = np.zeros([tvecs_shape[0]])
            for i in range(tvecs_shape[0]):
                [x, y, z] = tvecs[i][0]
                [t1, t2, t3] = rvecs[i][0]
                msg = str([ids[i], x, y, z, t1, t2, t3])
                arr = bytes(msg, 'utf-8')
                matlab_socket.send(arr)

    if recording:
        out.write(frame)
        cv2.putText(frame,'RECORDING',(0,30), font, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('Camera View (\'q\' to quit, \'c\' to toggle crosshairs, \'r\' to record)', frame)

    char = cv2.waitKey(1) & 0xFF
    if char == ord('q'):
        break
    elif char == ord('c'):
        crosshairs = not crosshairs
    elif char == ord('r'):
        recording = not recording
        

cap.release()
out.release()
cv2.destroyAllWindows()
