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


handImageSet = cv2.imread("Hands.png")
h, w, c = handImageSet.shape
HandImageSize = (85, 256)
HandImages = []
for i in range(0, 3):
    Image = handImageSet[0:h, i*w//3:(i+1)*w//3]
    Image = cv2.resize(Image, HandImageSize)
    HandImages.append(Image)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
handDetectorLeft = htm.handDetector(maxHands=1, detectionCon=0.5)
handDetectorRight = htm.handDetector(maxHands=1, detectionCon=0.5)

font = cv2.FONT_HERSHEY_PLAIN
tipIds = [4, 8, 12, 16, 20]
TimeLeft = 300
player1Status = "A"
player2Status = "A"
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    #SOL VE SAĞ TARAF
    imgLeft = img[0:h, 0:w//2]
    imgRight = img[0:h, w//2:w]

    # OYUN EKRANI / SÜRE
    cv2.line(img, (w//2, 50), (w//2, h), (0, 0, 255), 2)
    cv2.putText(img, "Player 1", (w//8, 30), font, 2, (0, 255, 0), 3)
    cv2.putText(img, "Player 2", (w//2 + w//8, 30), font, 2, (0, 255, 0), 3)
    cv2.putText(img, str(TimeLeft//100), (w//2 - 17, 40), font, 3.5, (255, 0, 0), 3)
    TimeLeft -= 3

    # 2 EKRANDA DA ELLERİN KONUMLARINI BULUYORUZ
    imgLeft = handDetectorLeft.findHands(imgLeft, False)
    imgRight = handDetectorRight.findHands(imgRight, False)
    handLeftPos = handDetectorLeft.findPosition(imgLeft)
    handRightPos = handDetectorRight.findPosition(imgRight)

    # SOL EKRAN TESPİTİ
    if len(handLeftPos) != 0:
        leftfingers = []
        for id in range(5):
            if handLeftPos[tipIds[id]][1] > handLeftPos[tipIds[id] - 2][1]:
                leftfingers.append(1)
            else:
                leftfingers.append(0)
        if sum(leftfingers) == 1:
            player1Status = "Taş"
            #img[0:HandImageSize[1], 0: HandImageSize[0]] = HandImages[0]
        elif sum(leftfingers) == 5:
            player1Status = "Kağıt"
            #img[0:HandImageSize[1], 0: HandImageSize[0]] = HandImages[1]
        else:
            player1Status = "Makas"
            #img[0:HandImageSize[1], 0: HandImageSize[0]] = HandImages[2]

    # SAĞ EKRAN TESPİTİ
    if len(handRightPos) != 0:
        rightfingers = []
        for id in range(5):
            if handRightPos[tipIds[id]][1] < handRightPos[tipIds[id] - 2][1]:
                rightfingers.append(1)
            else:
                rightfingers.append(0)
        if sum(rightfingers) == 1:
            player2Status = "Taş"
            #img[0: HandImageSize[1], w-HandImageSize[0]:w] = HandImages[0]
        elif sum(rightfingers) == 5:
            player2Status = "Kağıt"
            #img[0: HandImageSize[1], w-HandImageSize[0]:w] = HandImages[1]
        else:
            player2Status = "Makas"
            #img[0: HandImageSize[1], w-HandImageSize[0]:w] = HandImages[2]

    if round((TimeLeft/100),1) == 0.5:
        break

    imageArray = ([imgLeft, imgRight])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Taş Kağıt Makas', img)
    cv2.waitKey(1)

if player1Status == player2Status:
    result = "BERABERE"
elif player1Status == "Taş" and player2Status == "Kağıt":
    result = "Player 2 Kazandi!"
elif player1Status == "Taş" and player2Status == "Makas":
    result = "Player 1 Kazandi!"
elif player1Status == "Kağıt" and player2Status == "Makas":
    result = "Player 2 Kazandi!"
elif player1Status == "Kağıt" and player2Status == "Taş":
    result = "Player 1 Kazandi!"
elif player1Status == "Makas" and player2Status == "Taş":
    result = "Player 2 Kazandi!"
elif player1Status == "Makas" and player2Status == "Kağıt":
    result = "Player 1 Kazandi!"

cv2.putText(img, result, (w//2, h//4), font, 3, (0, 0, 0), 6)
cv2.imshow("Taş Kağıt Makas", img)
#cv2.imshow("Hand Images", Hand0)
cv2.waitKey()
cv2.destroyAllWindows()

