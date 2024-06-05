import cv2
video = cv2.VideoCapture(r"C:\Users\pushpak\Downloads\Untitled video - Made with Clipchamp.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 2)
frame_num = 0

while True:
    ret, frame = video.read()
    if frame_num % frame_interval == 0:
        img_name = f'C:\\Users\\pushpak\\pipeline_dataset\q\frame{frame_num}.jpg'
        cv2.imwrite(img_name, frame)
        print(f'Saved {img_name}')
    frame_num += 1
video.release()
