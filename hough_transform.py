import cv2
import numpy as np

# Load an image
img = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-23 012559.png", 0)
original = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-23 012559.png")

# Equalize the histogram
equ = cv2.equalizeHist(img)
gblur = cv2.GaussianBlur(equ, (3, 3), 0)

edges = cv2.Canny(gblur,20, 50, apertureSize=3)

# Perform a Hough Transform for line detection
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

# Iterate over the lines and draw them on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(original, (x1,y1), (x2,y2), (0,0,255), 2)

# Save the image, replace '/path/for/output/image.jpg' with the actual path
# cv2.imwrite('/path/for/output/image.jpg', img)

# Alternatively, to display the image, uncomment the two lines below

cv2.imshow("Detected Lines", original)
cv2.imshow("Edges",edges)
cv2.waitKey(0)