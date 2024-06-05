import cv2
import numpy as np

# Load image
img = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-22 045915.png", cv2.IMREAD_GRAYSCALE)
copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for later display
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
sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
edge_normalized = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
_, edge_thresh = cv2.threshold(edge_normalized, 15, 100, cv2.THRESH_BINARY)

# Apply dilation to enhance the edges
kernel = np.ones((3,3),np.uint8)
edge_dilated = cv2.dilate(edge_thresh, kernel, iterations = 1)

# Apply Hough line transform
lines = cv2.HoughLines(edge_dilated, rho=1, theta=np.pi/180, threshold=100)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 5000 * (-b))
        y1 = int(y0 + 5000 * (a))
        x2 = int(x0 - 5000 * (-b))
        y2 = int(y0 - 5000 * (a))

        cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Original with lines", copy)
cv2.waitKey(0)
cv2.destroyAllWindows()