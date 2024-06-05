import cv2
import numpy as np

def calculate_saturation(se, sw):
    return se / float(sw)

def onlyYellow(img, lower_yellow, upper_yellow):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_hue_range_low = np.array([lower_yellow, 50, 60])
    yellow_hue_range_high = np.array([upper_yellow, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, yellow_hue_range_low, yellow_hue_range_high)

    hue_channel = cv2.bitwise_and(hsv_img[:, :, 0], hsv_img[:, :, 0], mask=yellow_mask)
    thresh = cv2.threshold(hue_channel,100, 255, cv2.THRESH_BINARY)[1]
    return thresh

def compute_saturation(se, sw):
    return se/sw

def filter_disruptor_edges(img, win_size, saturation_threshold):
    height, width = img.shape
    edges = cv2.Canny(img, 100, 200)

    for i in range(0, height, win_size):
        for j in range(0, width, win_size):
            window = edges[i:i+win_size, j:j+win_size]
            se = np.count_nonzero(window)
            sw = win_size**2
            saturation = compute_saturation(se, sw)

            if saturation > saturation_threshold:
                edges[i:i+win_size, j:j+win_size] = 0

cap = cv2.VideoCapture(r"C:\Users\pushpak\Downloads\1701 12.06.2023 18.mp4")
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

scale = 1
delta = 0
ddepth = cv2.CV_16S
# Create a structuring element (you can change the size of kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
win_size = 20
saturation_threshold = 10

# Perform dilation
cv2.namedWindow("frame")

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a median blur to reduce noise 
        median_filtered = cv2.medianBlur(frame,3)
        edges = onlyYellow(median_filtered,20,360)

        dilation = cv2.dilate(edges, kernel, iterations = 1)

        # print(filtered_edges)

        # gblr = cv2.GaussianBlur(dilation, (3, 3),1)

        # grad_x = cv2.Sobel(gblr, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # grad_y = cv2.Sobel(gblr, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        # abs_grad_x = cv2.convertScaleAbs(grad_x)
        # abs_grad_y = cv2.convertScaleAbs(grad_y)
        # result_final = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        


        # Apply adaptive thresholding
        # thresh = cv2.adaptiveThreshold(median_filtered,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

        # # Apply morphological operations
        # kernel = np.ones((5,5), np.uint8)
        # morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        
        # Apply Canny Edge Detection
        filtered_edges = cv2.Canny(dilation, 100, 200)
        # f_edges = filter_disruptor_edges(filtered_edges, win_size, saturation_threshold)

        # Filter edges based on saturation
        # filtered_edges = np.zeros_like(edges)
        # window_size = 10
        # saturation_threshold = 30
        # for i in range(0, edges.shape[0] - window_size, window_size):
        #     for j in range(0, edges.shape[1] - window_size, window_size):
        #         window = edges[i:i+window_size, j:j+window_size]
        #         se = np.sum(window) / 255  
        #         sat = calculate_saturation(se, window_size * window_size)
                
        #         if sat * 50 < saturation_threshold:
        #             filtered_edges[i:i+window_size, j:j+window_size] = window
                    

        # lines = cv2.HoughLines(filtered_edges, 1, np.pi / 180, 100)
        # if lines is not None:
        #     for line in lines:
        #         rho, theta = line[0]
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * (a))
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * (a))

        #         cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.imshow('frame', dilation)
        # cv2.imshow("real",frame)
        # cv2.resizeWindow("frame",640,480)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()