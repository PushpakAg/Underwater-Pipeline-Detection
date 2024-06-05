import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-23 012559.png", 0)

# Equalize the histogram
equ = cv2.equalizeHist(img)
scale = 1
delta = 0
ddepth = cv2.CV_16S

# Apply Gaussian Blur
img = cv2.GaussianBlur(equ, (3, 3), 0)

# Apply Sobel operator
grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Calculate absolute values and perform normalization
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
edge = cv2.addWeighted(abs_grad_x,0.6, abs_grad_y, 0.6,0)
edge_normalized = cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX)
_, edge_thresh = cv2.threshold(edge_normalized, 40, 200, cv2.THRESH_BINARY)

# Apply dilation to enhance the edges
kernel = np.ones((3,3),np.uint8)
edge_dilated = cv2.dilate(edge_thresh, kernel, iterations = 1)

# lines = cv2.HoughLines(edge_dilated, 1, np.pi/180, 100)

# # Iterate over the lines and draw them on the original image
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
edges = cv2.Canny(edge_dilated,100, 200, apertureSize=3)

gblur = cv2.GaussianBlur(edges, (3, 3), 0)

cv2.imshow("Sobel edges", edge_dilated)
cv2.imshow("Detected Edges",gblur)
cv2.waitKey(0)
cv2.destroyAllWindows()