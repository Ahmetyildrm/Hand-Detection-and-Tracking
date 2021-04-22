import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import dlib

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

cap = cv2.VideoCapture("facialGestures.mp4")

handSkeleton = htm.handDetector(detectionCon=0.7, trackCon=0.7)
faceSkeleton = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
bg = np.zeros((480, 640, 3), np.uint8)
skeletonColor = (222, 202, 176)


while True:
    success, img = cap.read()
    #img = cv2.flip(img, 1)
    handedImage = handSkeleton.findHands(img.copy(), True)
    lmList = handSkeleton.findPosition(img)
    skeletonImage = handedImage - img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceSkeleton(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(0, 67):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            x1 = face_landmarks.part(n+1).x
            y1 = face_landmarks.part(n+1).y
            cv2.circle(skeletonImage, (x, y), 1, (0, 255, 255), 1)
            if n!=16 and n!=21 and n!= 26 and n!= 30 and n!= 35 and n!=41 and n!=47 and n!= 59 and n!= 67:
                cv2.line(skeletonImage, (x, y), (x1, y1), (0, 255, 0), 1)
        cv2.line(skeletonImage, (face_landmarks.part(67).x, face_landmarks.part(67).y), (face_landmarks.part(60).x, face_landmarks.part(60).y), (0, 255, 0), 1)
        cv2.line(skeletonImage, (face_landmarks.part(59).x, face_landmarks.part(59).y), (face_landmarks.part(48).x, face_landmarks.part(48).y), (0, 255, 0), 1)
        cv2.line(skeletonImage, (face_landmarks.part(36).x, face_landmarks.part(36).y), (face_landmarks.part(41).x, face_landmarks.part(41).y), (0, 255, 0), 1)
        cv2.line(skeletonImage, (face_landmarks.part(42).x, face_landmarks.part(42).y), (face_landmarks.part(47).x, face_landmarks.part(47).y), (0, 255, 0), 1)

        lefteyeX = 0
        lefteyeY = 0
        for n in range(36, 42):
            lefteyeX += face_landmarks.part(n).x
            lefteyeY += face_landmarks.part(n).y
        righteyeX = 0
        righteyeY = 0
        for n in range(42, 48):
            righteyeX += face_landmarks.part(n).x
            righteyeY += face_landmarks.part(n).y
        cv2.circle(skeletonImage, (lefteyeX//6, lefteyeY//6), 4, (29, 101, 181), -1)
        cv2.circle(skeletonImage, (righteyeX//6, righteyeY//6), 4, (29, 101, 181), -1)

    imageArray = ([img, skeletonImage])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)
    cv2.waitKey(1)