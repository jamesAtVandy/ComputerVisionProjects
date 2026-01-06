import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm

#######################
# Configuration
#######################
brushThickness = 15
eraserThickness = 50

# Color palette for cycling (Pink, Blue, Green)
colorPalette = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
colorIndex = 0
drawColor = colorPalette[colorIndex]  # Default pink

# Fist detection state for color cycling
fistDebounce = False
lastFistTime = 0

#######################
# Header Setup - Color Selection Bar
#######################
# Create header with color selection buttons
header = np.zeros((125, 1280, 3), np.uint8)
header[:] = (50, 50, 50)  # Dark gray background

# Define color buttons in the header
# Pink button (0-213)
cv2.rectangle(header, (0, 0), (213, 125), (255, 0, 255), cv2.FILLED)
cv2.putText(header, "PINK", (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Blue button (213-426)
cv2.rectangle(header, (213, 0), (426, 125), (255, 0, 0), cv2.FILLED)
cv2.putText(header, "BLUE", (280, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Green button (426-639)
cv2.rectangle(header, (426, 0), (639, 125), (0, 255, 0), cv2.FILLED)
cv2.putText(header, "GREEN", (480, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Title area (639-1067)
cv2.putText(header, "AI Virtual Painter", (700, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

# Eraser button (1067-1280)
cv2.rectangle(header, (1067, 0), (1280, 125), (0, 0, 0), cv2.FILLED)
cv2.putText(header, "ERASER", (1100, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#######################
# Camera Setup
#######################
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#######################
# Hand Detector
#######################
detector = htm.HandDetector(detection_con=0.85)

#######################
# Canvas for Drawing
#######################
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Previous position for drawing lines
xp, yp = 0, 0

#######################
# Main Loop
#######################
while True:
    # 1. Import image
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)
    
    # 2. Find Hand Landmarks
    hands, img = detector.find_hands(img)
    
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        
        # Tip of index finger for drawing position
        x1, y1 = lmList[8][0], lmList[8][1]  # Index finger tip

        # 3. Check which fingers are up using detector method
        fingers = detector.fingers_up(hand)
        # fingers = [thumb, index, middle, ring, pinky] (0 or 1 for each)
        
        totalFingersUp = sum(fingers)
        thumbUp = fingers[0]
        indexUp = fingers[1]
        middleUp = fingers[2]

        # 4. Closed Fist = Color Switching (no fingers up, hand detected)
        if totalFingersUp == 0:
            if not fistDebounce and (time.time() - lastFistTime) > 0.5:
                colorIndex = (colorIndex + 1) % len(colorPalette)
                drawColor = colorPalette[colorIndex]
                fistDebounce = True
                lastFistTime = time.time()
            xp, yp = 0, 0  # Reset position
            cv2.putText(img, "COLOR SWITCH", (x1-50, y1-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, drawColor, 2)
        else:
            fistDebounce = False

        # 5. Eraser Mode - Thumb + Index + Middle fingers up
        if thumbUp and indexUp and middleUp:
            cv2.circle(img, (x1, y1), eraserThickness//2, (128, 128, 128), cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            cv2.line(img, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
            xp, yp = x1, y1

        # 6. Pen Lift Mode - Thumb present (stops drawing)
        elif thumbUp:
            xp, yp = 0, 0  # Reset position to prevent connecting lines
            cv2.putText(img, "PEN UP", (x1-30, y1-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 7. Drawing Mode - Only Index finger up (no thumb)
        elif indexUp and not thumbUp:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
        
        else:
            # Any other gesture - reset drawing position
            xp, yp = 0, 0

    # 6. Merge canvas with camera image
    # Create grayscale version of canvas
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # Invert to create mask (white where we drew)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # Remove the drawing area from original image
    img = cv2.bitwise_and(img, imgInv)
    # Add the canvas drawing to the image
    img = cv2.bitwise_or(img, imgCanvas)

    # 7. Set header image
    img[0:125, 0:1280] = header
    
    # 8. Display
    cv2.imshow("AI Virtual Painter", img)
    cv2.imshow("Canvas", imgCanvas)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
