import cv2 as cv
import numpy as np
import os
import mediapipe as mp

# All the colors are inverted

# importing mediapipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5, max_num_hands = 1)

cap = cv.VideoCapture(0)

# Making the window
ret, frame = cap.read()
window = np.zeros((frame.shape[0], frame.shape[1], 3), dtype = 'uint8') + 255

# initialising required variables
prev_x = 0
prev_y = 0
color = (255, 255, 0)
R = 10
radius = R

# openCV camera feed
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    
    # detection
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Rendering hands on screen
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Getting coordinates of index finger tip and middle finger tip
            indexTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middleTip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            width, height = window.shape[1], window.shape[0]
            indexTip_x = (indexTip.x * width) // 1
            indexTip_y = (indexTip.y * height) // 1

            middle_x = (middleTip.x * width) // 1
            middle_y = (middleTip.y * height) // 1

            # If not started painting, initialise the previous pointers 
            if prev_x == 0 and prev_y == 0 :
                prev_x = indexTip_x
                prev_y = indexTip_y
            
            # Drawing Mode
            if (abs(indexTip_x - middle_x) > 40) or (abs(indexTip_y - middle_y) > 40):
                cv.line(window, (int(prev_x), int(prev_y)), (int(indexTip_x), int(indexTip_y)), color, thickness = radius)
            # Selection Mode
            else:
                # Red color selection
                if(indexTip_x >= 40 and indexTip_x < 160 and indexTip_y <= 40):
                    color = (255, 255, 0)
                    radius = R
                # Green color selection
                elif(indexTip_x >= 180 and indexTip_x < 300 and indexTip_y <= 40):
                    color = (255, 0, 255)
                    radius = R
                # Blue color selection
                elif(indexTip_x >= 320 and indexTip_x < 440 and indexTip_y <= 40):
                    color = (0, 255, 255)
                    radius = R
                # Eraser selection
                elif(indexTip_x >= 460 and indexTip_x < 580 and indexTip_y <= 40):
                    color = (255, 255, 255)
                    radius = 40
                # Clear the canvas
                elif(indexTip_x >= 0 and indexTip_x < frame.shape[0] - 1 and indexTip_y > 440 and indexTip_y <= image.shape[0]-1):
                    window = np.zeros((frame.shape[0], frame.shape[1], 3), dtype = 'uint8') + 255
            
            # Updating the previous coordinates
            prev_x = indexTip_x
            prev_y = indexTip_y
    else:
        # Initialising the previous points to (0, 0) again if no hand detected
        prev_x = 0
        prev_y = 0

    # Display on video logic
    # # Creating the mask with painted part as black
    gray = cv.cvtColor(window, cv.COLOR_BGR2GRAY)
    thresh, gray_mask = cv.threshold(gray, 240, 255, cv.THRESH_BINARY)
    gray_mask = cv.cvtColor(gray_mask, cv.COLOR_GRAY2BGR)
    # # Making painted part of video black
    image = cv.bitwise_and(image, gray_mask)
    # # Taking or with only painted part of canvas
    image = cv.bitwise_or(~(window), image)

    # Adding the UI to video
    image = cv.rectangle(image, (0, 0), (image.shape[1]-1, 60), (40, 40, 40), -1)
    image = cv.rectangle(image, (0, 420), (image.shape[1]-1, image.shape[0]-1), (200, 200, 200), -1)
    image = cv.rectangle(image, (40, 0), (160, 40), (0, 0, 255), -1)
    image = cv.rectangle(image, (180, 0), (300, 40), (0, 255, 0), -1)
    image = cv.rectangle(image, (320, 0), (440, 40), (255, 0, 0), -1)
    image = cv.rectangle(image, (460, 0), (580, 40), (255, 255, 255), -1)

    image = cv.putText(image, "CLEAR", (280, 460), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

    # cv.imshow('mask', gray_mask)
    # cv.imshow('gray', gray)
    cv.imshow('HandTracking', image)
    # cv.imshow('Canvas', window)
    # cv.imshow('NCanvas',~(window))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
       
cap.release()
cv.destroyAllWindows()