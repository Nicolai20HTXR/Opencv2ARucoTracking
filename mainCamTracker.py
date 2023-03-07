import cv2
import math
import numpy as np

# https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

npzfile = np.load("webcam_calibration_ouput_2.npz")
cameraMatrixData = npzfile["mtx"]
cameraDistData = npzfile["dist"]

cap = cv2.VideoCapture(0)


def findArcuco(img, marker_size=5, total_markers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f"DICT_{marker_size}X{marker_size}_{total_markers}")
    dictionary = cv2.aruco.getPredefinedDictionary(key)
    parameters = cv2.aruco.DetectorParameters()
    # parameters.useAruco3Detection = True       #https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html#a5ba4261aa9c097f48e4e64426b16a7e1
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    bbox, ids, _ = detector.detectMarkers(gray)


    if draw and ids is not None:
        objectMarker = cv2.aruco.drawDetectedMarkers(img, bbox, ids)
        for i, corners in enumerate(bbox):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, cameraMatrixData, cameraDistData
            )
            if rvec is not None:
                cv2.drawFrameAxes(
                    img, cameraMatrixData, cameraDistData, rvec, tvec, 5, 5
                )

        # For single:
        # rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(bbox, marker_size, cameraMatrixData, cameraDistData)
        # objectMarker = cv2.aruco.drawDetectedMarkers(img,bbox,ids)

        # if rvec is not None:
        #     cv2.drawFrameAxes(img, cameraMatrixData,cameraDistData,rvec,tvec,5,5)


while 1:
    _, frame = cap.read()
    findArcuco(frame, 4, 50)
    if cv2.waitKey(1) == 113:
        break
    cv2.imshow("Test", frame)
