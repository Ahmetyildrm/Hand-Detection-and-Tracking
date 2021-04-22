import cv2
import math
import mediapipe as mp
import time
import HandTrackingModule as htm
import random
import numpy as np
import matplotlib.pyplot as plt


## Stack all images in one image
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


def distBetweenPoints(point1, point2, lmList):
    x1, y1 = lmList[point1][1], lmList[point1][2]
    x2, y2 = lmList[point2][1], lmList[point2][2]
    return (point1, point2, math.hypot(x2 - x1, y2 - y1))


def printDistances(lmList):
    print("\n-----------------------------")
    print("Baş Parmak: ", distBetweenPoints(2, 4, lmList)[2])
    print("İşaret Parmak: ", distBetweenPoints(5, 8, lmList)[2])
    print("Orta Parmak: ", distBetweenPoints(9, 12, lmList)[2])
    print("Yüzük Parmak: ", distBetweenPoints(13, 16, lmList)[2])
    print("Serçe Parmak: ", distBetweenPoints(17, 20, lmList)[2])


def parmakUcuPos(lmList, fingerNo):
    return (lmList[fingerNo][1], lmList[fingerNo][2])

def findArea(img, X, Y, n):
    # Initialze area
    area = 0.0

    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0, n):
        area += (X[j] + X[i]) * (Y[j] - Y[i])
        j = i  # j is previous vertex to i

    # Return absolute value
    Area = int(abs(area / 2.0))
    cv2.putText(img, "A: " + str(round(Area, 2)), (int(sum(X)/len(X)), int(sum(Y)/len(Y))),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 2)

detector = htm.handDetector(maxHands=1)
blank_image = np.zeros((1328, 747, 3), np.uint8)
img = cv2.imread("SolOn.jpeg")

img = detector.findHands(img, True)
detector.findFingertips(img, 100)
lmList = detector.findPosition(img)

parmakUclari = [parmakUcuPos(lmList, 0), parmakUcuPos(lmList, 4), parmakUcuPos(lmList, 8), parmakUcuPos(lmList, 12),
                parmakUcuPos(lmList, 16), parmakUcuPos(lmList, 20)]

maxX = max(item[0] for item in parmakUclari)
minX = min(item[0] for item in parmakUclari)
maxY = max(item[1] for item in parmakUclari)
minY = min(item[1] for item in parmakUclari)

printDistances(lmList)

distances = []
for i in range(5, len(lmList) - 4):
    distance = distBetweenPoints(i, i + 4, lmList)[2]
    p1 = (lmList[i][1], lmList[i][2])
    p2 = (lmList[i + 4][1], lmList[i + 4][2])
    cv2.line(img, p1, p2, (255, 0, 0), 3)
    cv2.putText(img, str(i) + "-)" + str(round(distance, 2)), ((p1[0] + p2[0]) // 2 - 25, (p1[1] + p2[1]) // 2 - 12),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 2)
    cv2.circle(img, ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), 1, (255, 0, 255), 2)
    distances.append(distance)

fingerLengths = [distBetweenPoints(2, 4, lmList)[2], distBetweenPoints(5, 8, lmList)[2],
                 distBetweenPoints(9, 12, lmList)[2], distBetweenPoints(13, 16, lmList)[2],
                 distBetweenPoints(17, 20, lmList)[2]]
norm = [float(i) / max(fingerLengths) for i in fingerLengths]




X = [lmList[0][1], lmList[5][1], lmList[9][1], lmList[13][1]]
Y = [lmList[0][2], lmList[5][2], lmList[9][2], lmList[13][2]]

findArea(img, X, Y, 4)

img_offset = 40
# img_bbox = cv2.rectangle(img_handpoints, (minX-img_offset, minY-img_offset), (maxX+img_offset, maxY+img_offset), (255, 0, 0), 3)
img_crop = img[minY - img_offset:maxY + img_offset, minX - img_offset:maxX + img_offset]
cv2.imshow('Hand', img_crop)
cv2.waitKey()
cv2.destroyAllWindows()
