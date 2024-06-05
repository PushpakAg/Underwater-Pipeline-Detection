import cv2
import numpy as np

# def calculate_saturation(se, sw):
#     return se / float(sw)

def onlyYellow(img, lower_yellow, upper_yellow):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_hue_range_low = np.array([lower_yellow, 100, 100])
    yellow_hue_range_high = np.array([upper_yellow, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, yellow_hue_range_low, yellow_hue_range_high)

    hue_channel = cv2.bitwise_and(hsv_img[:, :, 0], hsv_img[:, :, 0], mask=yellow_mask)
    thresh = cv2.threshold(hue_channel, 90, 255, cv2.THRESH_BINARY)[1]
    return thresh

# Load an image
img_path = r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-02-28 202719.png"
original = cv2.imread(img_path)
printimg = original.copy()

edges = onlyYellow(printimg, 20, 150)
cv2.imshow("edges",edges)

# Define window size and saturation threshold
# window_size = 10
# saturation_threshold = 20

# Create an empty copy of the edge image to hold the final result
# filtered_edges = np.zeros_like(edges)

# # Scan image with window
# for i in range(0, edges.shape[0] - window_size, window_size):
#     for j in range(0, edges.shape[1] - window_size, window_size):
#         window = edges[i:i+window_size, j:j+window_size]
#         se = np.sum(window) / 255
#         sat = calculate_saturation(se, window_size * window_size)

#         if sat * 50 < saturation_threshold:
#             filtered_edges[i:i+window_size, j:j+window_size] = window
filtered_edges = cv2.Canny(edges,100,200)
cv2.imshow("filtered_edges",filtered_edges)

lines = cv2.HoughLines(filtered_edges, 1, np.pi / 180, 50)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(printimg, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image
# cv2.imshow("original", original)
cv2.imshow("result", printimg)
cv2.waitKey(0)
cv2.destroyAllWindows()