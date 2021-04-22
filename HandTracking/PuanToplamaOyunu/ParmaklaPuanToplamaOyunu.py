import cv2
import math
import mediapipe as mp
import time
import HandTrackingModule as htm
import random
import urllib.request
import numpy as np



def randomColor():
    b = random.randint(1, 255)
    g = random.randint(1, 255)
    r = random.randint(1, 255)
    return (b, g, r)


def dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.hypot(x2 - x1, y2 - y1)


def randomPoint(img, lmList, rX, rY, score, fingerType):
    h, w, c = img.shape
    isaretX, isaretY = lmList[fingerType][1], lmList[fingerType][2]
    distance = dist((isaretX, isaretY), (rX, rY))
    if distance <= 10:
        rX, rY = random.randint(1, w), random.randint(1, h // 2)
        score += 1
    return rX, rY, score


cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1)
rX, rY = random.randint(1, 200), random.randint(1, 200 // 2)
score = 0
startTime = time.time()

fingerType = 8
maxScore = 8

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img, False)
    lmList = detector.findPosition(img)
    detector.findFingertips(img, fingerType)

    if len(lmList) != 0:
        rX, rY, score = randomPoint(img, lmList, rX, rY, score, fingerType)

    cv2.circle(img, (rX, rY), 15, (255, 0, 0), cv2.FILLED)
    cTime = time.time()

    cv2.putText(img, "Score: " + str(score), (500, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.putText(img, "Time: " + str(round((cTime - startTime), 3)), (10, 45), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (255, 255, 100), 2)

    if score == maxScore:
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)

cv2.imshow("Image", img)
cv2.waitKey()
cv2.destroyAllWindows()
