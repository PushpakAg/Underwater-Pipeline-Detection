import cv2
import numpy as np

def onlyYellow(img, lower_yellow, upper_yellow):
    b, g, r = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25,25))

    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    cl_img = cv2.merge([b, g, r])

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_hue_range_low = np.array([lower_yellow, 130, 90])
    yellow_hue_range_high = np.array([upper_yellow, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, yellow_hue_range_low, yellow_hue_range_high)
    

    hue_channel = cv2.bitwise_and(hsv_img[:, :, 0], hsv_img[:, :, 0], mask=yellow_mask)
    thresh, mask = cv2.threshold(hue_channel,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def polyfit_sliding_window(binary, pipe_width_px=30, visualise=True, diagnostics=False):
    if binary is None or binary.max() == 0:
        return np.zeros_like(binary)
    
    out = np.dstack((binary, binary, binary))

    nonzero = binary.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # Fit a second-degree polynomial to the nonzero pixels
    left_fit_coef = np.polyfit(nonzeroy, nonzerox, 2)
    
    # Generate the x and y values for the polynomial curve
    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
    fitx = left_fit_coef[0]*ploty**2 + left_fit_coef[1]*ploty + left_fit_coef[2]
    fitx = fitx.astype(np.int32)
    
    # Draw the polynomial curve on the output image
    line_image = np.zeros_like(out)
    for i in range(len(ploty)):
        cv2.line(line_image, (fitx[i], int(ploty[i])), (fitx[i], int(ploty[i])), (255, 0, 255), 2)

    return line_image

cap = cv2.VideoCapture(r"C:\Users\pushpak\Downloads\Untitled video - Made with Clipchamp (3).mp4")

while True :
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
    mask = onlyYellow(frame,20,150)

    mask_resized = cv2.resize(mask,(640,480))
    # mask = np.asarray(mask)
    # mask = cv2.cvtColor(cv2.COLOR_BGR2GRAY)
    line_image = polyfit_sliding_window(mask)

    line_resized = cv2.resize(line_image,(640,480))

    cv2.imshow("original",mask_resized)

    cv2.imshow("line",line_resized)
    cv2.moveWindow("line",0,0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
