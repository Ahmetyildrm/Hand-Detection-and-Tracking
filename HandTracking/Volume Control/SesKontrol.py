import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver


def distBetweenPoints(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    return math.hypot(x2 - x1, y2 - y1)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))

font = cv2.FONT_HERSHEY_PLAIN

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
volumeController = htm.handDetector(maxHands=1, detectionCon=0.8, trackCon=0.7)
bg = np.zeros((480, 640, 3), np.uint8)
newVolume = -25
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    handedImage = volumeController.findHands(img.copy(), False)
    lmList = volumeController.findPosition(img)

    cv2.rectangle(handedImage, (10, 10), (40, 300), (0, 255, 0), 4)

    if len(lmList) != 0:
        basParmak = (lmList[4][1], lmList[4][2])
        isaretParmak = (lmList[8][1], lmList[8][2])
        cv2.circle(handedImage, basParmak, 8, (0, 0, 255), -1)
        cv2.circle(handedImage, isaretParmak, 8, (0, 0, 255), -1)
        cv2.circle(handedImage, basParmak, 10, (255, 255, 255), 2)
        cv2.circle(handedImage, isaretParmak, 10, (255, 255, 255), 2)

        cv2.line(handedImage, basParmak, isaretParmak, (255, 0, 0), 2)
        distance = distBetweenPoints(basParmak, isaretParmak)
        center = ((basParmak[0] + isaretParmak[0]) // 2, (basParmak[1] + isaretParmak[1]) // 2)
        cv2.circle(handedImage, center, 5, (255, 0, 255), -1)

        if distance <= 15:
            cv2.circle(handedImage, center, 5, (0, 255, 0), -1)

        distancePerc = distance * 100 / 22

        if distance > 150:
            distance = 150

        newVolume = np.interp(distance, [15, 150], [-62.5, 0])
        newVolume2 = np.interp(newVolume, [-62.5, 0], [0, 100])
        cv2.putText(handedImage, str(int(newVolume2)), (15, 325), font, 1.2, (0, 0, 0), 3)

        volumebarPerc = np.interp(distance, [15, 150], [0, 290])
        cv2.rectangle(handedImage, (35, 300), (15, 300 - int(volumebarPerc)), (255, 255, 0), -1)
    volume.SetMasterVolumeLevel(newVolume, None)

    # -------------------------------------------#
    skeletonImage = handedImage - img
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 25), font, 1.5, (255, 255, 100), 2)

    # imageArray = ([img, handedImage, skeletonImage],
    #               [bg, bg, bg])
    # stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', handedImage)
    cv2.waitKey(1)
