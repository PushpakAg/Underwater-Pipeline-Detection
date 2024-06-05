import cv2
import numpy as np

def preprocess(frame):
    try:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25,25))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        processed_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Apply median filtering for denoising
        processed_frame = cv2.medianBlur(processed_frame, 5)

        return processed_frame
    except cv2.error as e:
        print(f"Error in preprocess: {e}")
        return frame

def convert_color_space(frame):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv
    except cv2.error as e:
        print(f"Error in convert_color_space: {e}")
        return frame

def threshold_image(frame):
    try:
        # Otsu's thresholding
        _, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    except cv2.error as e:
        print(f"Error in threshold_image: {e}")
        return frame

def apply_morphology(thresh):
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return morph
    except cv2.error as e:
        print(f"Error in apply_morphology: {e}")
        return thresh

def detect_aruco_markers(morph_frame):
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(morph_frame, aruco_dict, parameters=parameters)
        return corners, ids
    except cv2.error as e:
        print(f"Error in detect_aruco_markers: {e}")
        return [], []

# Read the video feed
cap = cv2.VideoCapture(r"C:\Users\pushpak\Downloads\1701 12.06.2023 18.mp4")

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Preprocessing
    processed_frame = preprocess(frame)

    # Step 2: Color Space Conversion
    hsv_frame = convert_color_space(processed_frame)

    # Step 3: Thresholding and Binarization
    thresh = threshold_image(hsv_frame[:, :, 2])  # Use the Value channel of HSV

    # Step 4: Morphological Operations
    morph_frame = apply_morphology(thresh)

    # Step 5: ArUco Marker Detection
    corners, ids = detect_aruco_markers(morph_frame)

    # Draw markers on the original frame
    marked_frame = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    cv2.imshow("Marked Frame", marked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()