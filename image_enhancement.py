import cv2

def CLAHE(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray_image)
    if len(image.shape) == 3:
        clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    return clahe_image

def white_balance(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    cl = clahe.apply(l)
    balanced_lab_image = cv2.merge((cl, a, b))
    balanced_image = cv2.cvtColor(balanced_lab_image, cv2.COLOR_LAB2BGR)
    return balanced_image

def Contrast_Up(image):
    contrasted_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    return contrasted_image

def Contrast_Down(image):
    contrasted_image = cv2.convertScaleAbs(image, alpha=0.2, beta=0)
    return contrasted_image

def Brightness_Up(image):
    brightened_image = cv2.convertScaleAbs(image, alpha=1.0, beta=150)
    return brightened_image

def Brightness_Down(image):
    darkened_image = cv2.convertScaleAbs(image, alpha=1.0, beta=10)
    return darkened_image

def enhance_image(image, inp):
    for ind in inp:
        if ind == 0:
            image = white_balance(image)
        elif ind == 1:
            image = Contrast_Up(image)
        elif ind == 2:
            image = Contrast_Down(image)
        elif ind == 3:
            image = Brightness_Up(image)
        elif ind == 4:
            image = Brightness_Down(image)
        elif ind == 5:
            image = CLAHE(image)
    return image

inp = [3, 3, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 5, 4]
while True:
    image = cv2.imread(r"C:\Users\pushpak\Pictures\Screenshots\Screenshot 2024-04-08 185334.png")
    enhanced_image = enhance_image(image,inp)
    contrast_up = Contrast_Down(image)
    cv2.imshow("Enhanced Image", contrast_up)
    if cv2.waitKey(0):
        break

cv2.destroyAllWindows()