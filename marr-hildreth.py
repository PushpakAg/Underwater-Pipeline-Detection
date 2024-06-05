import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-22 045915.png", cv2.IMREAD_GRAYSCALE)

# Apply gaussian blur
image_blur = cv2.GaussianBlur(image,(4,4), 0)

# Compute Laplacian of the image
laplace = cv2.Laplacian(image_blur, cv2.CV_64F)

# Convert float values to absolute values
laplace = np.absolute(laplace)

# Normalize the result
laplace_norm = cv2.normalize(laplace, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Apply a binary threshold
_, binary_image = cv2.threshold(laplace_norm, 50, 255, cv2.THRESH_BINARY)

# Display the result
cv2.imshow("Marr-Hildreth", binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()