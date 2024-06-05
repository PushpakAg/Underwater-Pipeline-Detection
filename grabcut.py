import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
original = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-22 045915.png")  # Replace with your image

# Create a mask with the same dimension as the image
mask = np.zeros(original.shape[:2],np.uint8)

# Create background and foreground models
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# Define the region of interest (ROI) for GrabCut to act on.
# You need to adjust these values based on the pipeline ROI in your image
rect = (50,50,450,290)   # Replace with pipeline ROI coordinates

# Apply GrabCut
cv2.grabCut(original, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Segment the pipeline
pipeline_segmented = original * mask2[:,:,np.newaxis]

# Display the result
cv2.imshow("Segmented Pipeline", pipeline_segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()