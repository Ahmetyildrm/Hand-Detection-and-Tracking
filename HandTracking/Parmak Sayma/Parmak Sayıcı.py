import cv2
import numpy as np
import time
import HandTrackingModule as htm


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

handImageSet = cv2.imread("Resimler/HandCountingSet.jpg")
h, w, c = handImageSet.shape
HandImageSize = (150, 150)
HandImages = []
for i in range(0, 3):
    Image = handImageSet[0:h//2, i*w//3:(i+1)*w//3]
    Image = cv2.resize(Image, HandImageSize)
    HandImages.append(Image)
for i in range(3):
    Image = handImageSet[h//2:h, i*w//3:(i+1)*w//3]
    Image = cv2.resize(Image, HandImageSize)
    HandImages.append(Image)

cap = cv2.VideoCapture(0)
handDetector = htm.handDetector(detectionCon=0.7)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = handDetector.findHands(img, False)
    lmList = handDetector.findPosition(img)

    if len(lmList) != 0:
        fingers = []
        for id in range(5):
            if id == 0:
                if lmList[tipIds[id]][1] > lmList[tipIds[id]-2][1]:
                    fingers.append(0)
                else:
                    fingers.append(1)
            elif lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        img[0:HandImageSize[1], 0:HandImageSize[0]] = HandImages[sum(fingers)]
        cv2.rectangle(img, (0, 200), (100, 280), (0, 255, 0), -1)
        cv2.putText(img, str(sum(fingers)), (35, 260), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)




    imageArray = ([HandImages[0], HandImages[1], HandImages[2]],
                  [HandImages[3], HandImages[4], HandImages[5]])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', img)

    #cv2.imshow("Hand Images", Hand0)
    cv2.waitKey(1)
