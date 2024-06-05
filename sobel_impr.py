import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\pushpak\Downloads\WhatsApp Image 2024-04-08 at 21.41.34.jpeg", cv2.IMREAD_GRAYSCALE)
copy = img.copy()
scale = 1
delta = 0
ddepth = cv2.CV_16S

# Apply Gaussian Blur
img = cv2.GaussianBlur(img, (3, 3), 0)

# Apply Sobel operator
grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Calculate absolute values and perform normalization
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
edge = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
edge_normalized = cv2.normalize(edge, None, 15, 255, cv2.NORM_MINMAX)
_, edge_thresh = cv2.threshold(edge_normalized,50,100, cv2.THRESH_BINARY)

# Apply dilation to enhance the edges
kernel = np.ones((3,3),np.uint8)
edge_dilated = cv2.dilate(edge_thresh, kernel, iterations = 1)


cv2.imshow("Sobel edges", edge_dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()