import cv2
import numpy as np

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
    # if visualise:
    #     plot_images([(binary, 'Zoomed Binary Edges'), (out, 'Polynomial Curve'), (line_image, 'Line Image')])

    return line_image

