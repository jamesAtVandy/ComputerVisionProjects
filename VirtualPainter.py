import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header=overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

while True:
    success, img = cap.read()
    cv2.waitKey(1)
