import cv2 as cv
import numpy as np
import os
import mediapipe as mp

# importing mediapipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5)

# openCV camera feed
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    
    # detection
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # print(results.multi_hand_landmarks)
    
    cv.imshow('HandTracking', image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
       
cap.release()
cv.destroyAllWindows()