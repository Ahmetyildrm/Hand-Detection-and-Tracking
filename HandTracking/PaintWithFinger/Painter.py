import cv2
import math
import mediapipe as mp
import time
import HandTrackingModule as htm
import random
import numpy as np

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

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
detector = htm.handDetector(detectionCon=0.7)
bg = np.zeros((height, width, 3), np.uint8)
bg_last = np.zeros((height, width, 3), np.uint8)
color = (0, 255, 0)
basla = False

tipIds = [8, 12, 16, 20]
fingers = [0, 0, 0, 0]
isaretParmakPos = (0, 0)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, False)
    lmList = detector.findPosition(img)


    if len(lmList) != 0:
        if len(lmList) == 21:
            isaretParmakPos = (lmList[8][1], lmList[8][2])
        if len(lmList) == 42:
            isaretParmakPos = (lmList[29][1], lmList[29][2])
        fingers = []
        for id in range(4):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        if basla == False:
            bg = bg_last.copy()
            cv2.circle(bg, isaretParmakPos, 5, (255, 255, 255), 2)
            cv2.circle(bg, isaretParmakPos, 3, (0, 0, 255), -1)

        if basla == True:
            bg = bg_last.copy()
            cv2.circle(bg, isaretParmakPos, 7, color, -1)
            bg_last = bg.copy()



    if cv2.waitKey(1) == ord("s") or sum(fingers) == 0:  # s bastığında silecek
        bg = np.zeros((height, width, 3), np.uint8)
        bg_last = np.zeros((height, width, 3), np.uint8)

    elif sum(fingers) == 1:  # b bastığında başlayacak
        basla = True

    elif sum(fingers) == 4:  # d bastığında duracak
        basla = False


    #  RENK PALETLERİ
    h, w, c = bg.shape
    greenColorPos = (w - 32, 25)
    blueColorPos = (w - 32, 95)
    redColorPos = (w - 32, 165)
    whiteColorPos = (w - 32, 235)
    cv2.rectangle(bg, (w - 50, 10), (w - 25, 40), (0, 255, 0), -1)
    cv2.rectangle(bg, (w - 50, 80), (w - 25, 110), (255, 0, 0), -1)
    cv2.rectangle(bg, (w - 50, 150), (w - 25, 180), (0, 0, 255), -1)
    cv2.rectangle(bg, (w - 50, 220), (w - 25, 250), (255, 255, 255), -1)

    if basla == False and color != (0, 255, 0) and distBetweenPoints(isaretParmakPos, greenColorPos) < 15:
        color = (0, 255, 0)
        cv2.circle(bg, greenColorPos, 7, (0, 0, 0), -1)
        print("Green Picked")

    if basla == False and color != (255, 0, 0) and distBetweenPoints(isaretParmakPos, blueColorPos) < 15:
        color = (255, 0, 0)
        cv2.circle(bg, blueColorPos, 7, (0, 0, 0), -1)
        print("Blue Picked")

    if basla == False and color != (0, 0, 255) and distBetweenPoints(isaretParmakPos, redColorPos) < 15:
        color = (0, 0, 255)
        cv2.circle(bg, redColorPos, 7, (0, 0, 0), -1)
        print("Red Picked")

    if basla == False and color != (255, 255, 255) and distBetweenPoints(isaretParmakPos, whiteColorPos) < 15:
        color = (255, 255, 255)
        cv2.circle(bg, whiteColorPos, 7, (0, 0, 0), -1)
        print("White Picked")


    imageArray = ([bg, img])
    stackedImage = stackImages(imageArray, 0.7)
    cv2.imshow('Stacked Images', stackedImage)
    cv2.waitKey(1)