import cv2
import numpy as np

# Reads an image
img = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-23 004613.png", 0) 
scale = 1
delta = 0
ddepth = cv2.CV_16S
# Create a structuring element (you can change the size of kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Perform dilation
dilation = cv2.dilate(img, kernel, iterations = 1)

# Perform erosion
erosion = cv2.erode(img, kernel, iterations = 1)

gblr = cv2.GaussianBlur(erosion, (3, 3), 0)

grad_x = cv2.Sobel(gblr, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gblr, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Generate output image
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
result_final = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

cv2.imshow("Gaussian Blur",gblr)
cv2.imshow("result",result_final)
# cv2.imshow('Original image', img)
# cv2.imshow('Dilated image', dilation)
# cv2.imshow('Eroded image', erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()