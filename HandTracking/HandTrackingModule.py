import cv2
import math
import mediapipe as mp
import time
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img):
        lmList = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
        return lmList

    def CemberveYazi(self, img, id, lmId, point, color, text):
        if id == lmId:
            cv2.circle(img, point, 5, color, 2)
            cv2.putText(img, text, (point[0] + 8, point[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    def findFingertips(self, img, fingerNo):
        lmList = self.findPosition(img)
        for id, cx, cy in lmList:
            if fingerNo == 100:
                self.CemberveYazi(img, id, 4, (cx, cy), (255, 255, 255), "Bas")
                self.CemberveYazi(img, id, 8, (cx, cy), (255, 255, 255), "Isaret")
                self.CemberveYazi(img, id, 12, (cx, cy), (255, 255, 255), "Orta")
                self.CemberveYazi(img, id, 16, (cx, cy), (255, 255, 255), "Yuzuk")
                self.CemberveYazi(img, id, 20, (cx, cy), (255, 255, 255), "Serce")
            else:
                self.CemberveYazi(img, id, fingerNo, (cx, cy), (255, 255, 255), "")

    def skeleton(self, img):
        handedImage = self.findHands(img.copy(), True)
        skeletonImage = handedImage - img

        return skeletonImage

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img, True)
        # lmList = detector.findPosition(img)
        detector.findFingertips(img, 100)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 100), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
