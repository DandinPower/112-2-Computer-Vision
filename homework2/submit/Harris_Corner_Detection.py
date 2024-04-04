import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import convolve

def gaussian_smooth(img:np.ndarray, kernel_size: int, sigma: int):
    ########################################################################
    # TODO:                                                                #
    #   Perform the Gaussian Smoothing                                     #
    #   Input: image, kernel window size, sigma                                          #
    #   Output: smoothing image                                            #
    ########################################################################
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel * kernel.T
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return convolve(img, kernel)

def sobel_edge_detection(gray_img: np.ndarray):
    ########################################################################
    # TODO:                                                                #
    #   Perform the sobel edge detection                                   #
    #   Input: image after smoothing                                       #
    #   Output: the magnitude and direction of gradient                    #
    ########################################################################

    GRADIENT_THRESHOLD = 25

    # Convert gray_img to int64
    gray_img_array = gray_img.astype(np.int64)

    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    ix = convolve(gray_img_array, sobel_x_kernel)
    iy = convolve(gray_img_array, sobel_y_kernel)

    # filter weak gradient
    ix[ix < GRADIENT_THRESHOLD] = 0
    iy[iy < GRADIENT_THRESHOLD] = 0

    gradient_magnitude = np.sqrt(np.square(ix) + np.square(iy))
    gradient_direction = np.arctan2(iy, ix)

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return  (gradient_magnitude, gradient_direction)


def structure_tensor(gradient_magnitude: np.ndarray, gradient_direction: np.ndarray, window_size: int, sigma: float, k: float):
    ########################################################################
    # TODO:                                                                #
    #   Perform the cornermess response                                    #
    #   Input: gradient_magnitude, gradient_direction                      #
    #   Output: corner response matrix                                          #
    ########################################################################

    ix = gradient_magnitude * np.cos(gradient_direction)
    iy = gradient_magnitude * np.sin(gradient_direction)

    ixx = np.square(ix)
    ixy = ix * iy 
    iyy = np.square(iy)

    ixx = gaussian_smooth(ixx, window_size, sigma)
    ixy = gaussian_smooth(ixy, window_size, sigma)
    iyy = gaussian_smooth(iyy, window_size, sigma)

    det = ixx * iyy - np.square(ixy)
    trace = ixx + iyy 

    response_matrix = det - k * np.square(trace)

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return  response_matrix

def NMS(response_matrix, threshold):
    ########################################################################
    # TODO:                                                                #
    #   Perform the Non-Maximum Suppression                                #
    #   Input: corner response matrix, window size, threshold                    #
    #   Output: filtered coordinators                                      #
    ########################################################################
    
    response_matrix = cv2.dilate(response_matrix, None)
    threshold_value = threshold * np.max(response_matrix)
    filtered_coords = np.zeros(response_matrix.shape, dtype=bool)
    for i in range(1, response_matrix.shape[0] - 1):
        for j in range(1, response_matrix.shape[1] - 1):
            if response_matrix[i, j] > threshold_value:
                local_max = response_matrix[i-1:i+2, j-1:j+2].max()
                if np.isclose(response_matrix[i, j], local_max):
                    filtered_coords[i, j] = True

    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    return filtered_coords
    
def rotate(image, angle, center = None, scale = 1.0):

    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated