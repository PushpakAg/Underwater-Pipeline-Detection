import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Load and preprocess the image
img = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-04-22 001628.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# Perform Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=10, maxLineGap=5)

# Create feature vector
feature_vectors = np.array([[line[0][0], line[0][1], line[0][2], line[0][3]] for line in lines])

# Determine epsilon and minPts (you may need to experiment with these values)
epsilon = 0.04
min_samples = 2

# Apply DBSCAN
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(feature_vectors)
labels = db.labels_

# Visualize clustered line segments
for i, line in enumerate(lines):

    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Clustered Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()