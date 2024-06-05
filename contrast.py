import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

def img_enhancer(img):
    b, g, r = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25,25))

    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    cl_img = cv2.merge([b, g, r])

    return cl_img

def onlyYellow(img, lower_yellow, upper_yellow):
    cl_img = img_enhancer(img)

    hsv_img = cv2.cvtColor(cl_img, cv2.COLOR_BGR2HSV)

    yellow_hue_range_low = np.array([lower_yellow, 100, 100])
    yellow_hue_range_high = np.array([upper_yellow, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, yellow_hue_range_low, yellow_hue_range_high)

    hue_channel = cv2.bitwise_and(hsv_img[:, :, 0], hsv_img[:, :, 0], mask=yellow_mask)
    thresh, mask = cv2.threshold(hue_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask,hsv_img

cap = cv2.VideoCapture(r"C:\Users\pushpak\Downloads\Untitled video - Made with Clipchamp (3).mp4")

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        median_filtered = cv2.medianBlur(frame, 3)
        impr_frame = img_enhancer(median_filtered)
        mask, hsv_img = onlyYellow(frame, 20, 150)

        frame_resized = cv2.resize(frame, (640, 360))
        mask_resized = cv2.resize(mask, (640, 360))
        impr_frame_resized = cv2.resize(impr_frame,(640,360))
        # hsv_resized = cv2.resize(hsv_img,(640,480))

        fd, hog_image = hog(impr_frame_resized, orientations=9, pixels_per_cell=(8,8),
                             cells_per_block=(2, 2),block_norm = "L1-sqrt", visualize=True,channel_axis = -1)

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        hog_image_uint8 = (hog_image_rescaled * 255).astype(np.uint8)

        combined = np.hstack((frame_resized, cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), cv2.cvtColor(hog_image_uint8, cv2.COLOR_GRAY2BGR)))


        cv2.imshow('Frame, Mask, and HOG', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
