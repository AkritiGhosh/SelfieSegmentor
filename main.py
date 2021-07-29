import cv2
import mediapipe as mp
import numpy as np

# Processing
from matplotlib import pyplot as plt
from matplotlib import gridspec as gds

mp_selfie = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)
# Create with statement for model 
with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Apply segmentation
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model.process(frame)
        frame.flags.writeable = True

        cv2.imshow('Selfie Seg', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


plt.figure(figsize=(15,15))
grid = gds.GridSpec(1,2) #(row, column)
ax0 = plt.subplot(grid[0])
ax1 = plt.subplot(grid[1])
ax0.imshow(frame)
ax1.imshow(res.segmentation_mask)
plt.show()

# CREATE BACKGROUND
background = np.zeros(frame.shape, dtype=np.uint8)
