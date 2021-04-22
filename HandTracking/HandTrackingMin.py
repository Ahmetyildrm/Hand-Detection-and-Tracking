import cv2
import math
import mediapipe as mp
import time

def CemberveYazi(img, lmId, point, color, text):
    if id == lmId:
        cv2.circle(img, point, 3, color, 5)
        cv2.putText(img, text, (point[0]+8, point[1]-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

def ParmaklariBelirt():
    CemberveYazi(img, 4, (cx, cy), (255, 255, 255), "Bas")
    CemberveYazi(img, 8, (cx, cy), (255, 255, 255), "Isaret")
    CemberveYazi(img, 12, (cx, cy), (255, 255, 255), "Orta")
    CemberveYazi(img, 16, (cx, cy), (255, 255, 255), "Yuzuk")
    CemberveYazi(img, 20, (cx, cy), (255, 255, 255), "Serce")

def writeDistance(img, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.hypot(x2-x1, y2-y1)
    textPoint = ((x2+x1)//2, (y2+y1)//2)
    cv2.putText(img, str(int(distance)), textPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #print("-------------------------")
            for id, lm in enumerate(handLms.landmark):
                #print(id,"\n", lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, " x:", cx, "y:", cy)
                ParmaklariBelirt()
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 100), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
