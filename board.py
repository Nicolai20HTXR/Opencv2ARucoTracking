# import cv2
# import numpy as np

# Testing board img
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# board = cv2.aruco.GridBoard((3,5), 0.09,0.09, aruco_dict)
# img = board.generateImage((988,1470),marginSize =50)
# print(type(img))
# cv2.imwrite('test.png',img)


# debugging npzfile
# npzfile = np.load('webcam_calibration_ouput_2.npz')
# print(npzfile['dist'])


import cv2
from datetime import datetime
import numpy as np

webcam = cv2.VideoCapture(0)

# https://docs.opencv.org/4.7.0/da/d0d/tutorial_camera_calibration_pattern.html
x = 9
y = 6

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((y * x, 3), np.float32)
objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
objpoints = []
imgpoints = []
i = 0

while i < 10:
    _, image = webcam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (x, y), None)
    print(ret)

    if ret == True:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        objpoints.append(objp)
        cv2.drawChessboardCorners(image, (x, y), corners, ret)
        i += 1

    cv2.imshow("grid", image)
    cv2.waitKey(1000)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
np.savez(
    "webcam_calibration_ouput_2", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs
)
