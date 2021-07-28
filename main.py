import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_selfie = mp.solutions.selfie_segmentation

with mp_selfie.SelfieSegmentation(model_selection = 0) as model:
    while cap.isOpened():
        _, img = cap.read()
        cv2.imshow("Selfie", img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()
exit()


