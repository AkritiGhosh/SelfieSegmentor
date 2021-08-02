import cv2
import mediapipe as mp
import numpy as np
import gradio as gr

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
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model.process(framergb)
        frame.flags.writeable = True
        background = np.zeros(frame.shape, dtype=np.uint8)
        # change dimension of mask from 2d to 3d=> having 3 color channels and only values above 0.5 are True, rest are False
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.38
        #  If mask[a][b][c] = True then frame[a][b][c], else background[a][b][c]
        # For black background
        segmented_image = np.where(mask, frame, background)

        # segmented_image = np.where(mask, frame, cv2.blur(frame, (40,40)))
        cv2.imshow('Selfie Seg', segmented_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# plt.figure(figsize=(15,15))
# grid = gds.GridSpec(1,2) #(row, column)
# ax0 = plt.subplot(grid[0])
# ax1 = plt.subplot(grid[1])
# ax0.imshow(segmented_image)
# ax1.imshow(res.segmentation_mask)
# plt.show()

# def segment(img):
#     with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
#             res = model.process(img)
#             mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5
#             return  np.where(mask, img, cv2.blur(img, (40,40)))


# webcam = gr.inputs.Image(shape = (640, 480), source="webcam") 

# # Create gradio interface
# webapp = gr.interface.Interface(fn = segment, inputs=webcam, outputs = "img")
# webapp.launch()