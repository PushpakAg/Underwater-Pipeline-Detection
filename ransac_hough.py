import cv2
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

cap = cv2.VideoCapture(r"C:\Users\pushpak\Downloads\WhatsApp Video 2024-04-21 at 22.53.39.mp4")

ransac = linear_model.RANSACRegressor(min_samples=1)

while(cap.isOpened()):
    # apply filters
    ret, frame = cap.read()
    median_filtered = cv2.medianBlur(frame,3)
    canny_edges = cv2.Canny(median_filtered,100,150)
    
    # Probabilistic Hough Line Transform
    linesP = cv2.HoughLinesP(canny_edges, 1, np.pi/180, 50, None, 50,5)

    if linesP is not None:
        lines_with_dist_angle = []
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            m = (l[3]-l[1]) / (l[2]-l[0]) # slope
            c = l[1] - m*l[0] # intercept
              
            try: 
                ransac.fit(np.array([[pt] for pt in [1, m]]), [c, c])
                inlier_mask = ransac.inlier_mask_
            
                # check if the line is not outlier
                if np.count_nonzero(inlier_mask) > 0:
                    cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2)

                    # get features angle and distance
                    angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180. / np.pi
                    dist = np.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2)

                    lines_with_dist_angle.extend([angle, dist])

            except ValueError:
                pass

        # DBSCAN, lines clustering
        lines_with_dist_angle = np.array(lines_with_dist_angle).reshape(-1, 2)

        scaler = StandardScaler()
        lines_with_dist_angle_scaled = scaler.fit_transform(lines_with_dist_angle)
        db = DBSCAN(eps=0.7, min_samples=20).fit(lines_with_dist_angle_scaled)

        colors = [int(i) for i in list(db.labels_)]
        for i in range(len(colors)):
            l = linesP[i][0]
            if(colors[i] != -1):  # check for outliers which marked as -1 by DBSCAN
                cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2)

        cv2.imshow('clustered lines', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()