import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import filters

from Harris_Corner_Detection import gaussian_smooth, sobel_edge_detection, structure_tensor, NMS, rotate

SIGMA=5
THRESHOLD=0.01
K=0.04
ANGLE=30

if __name__ == '__main__':
    img_path = os.path.join('./original.jpg')
    img = cv2.imread(img_path)
    img_copy = np.copy(img)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # A. Harris corner detection with original image
    # i. Gaussian smooth results: 𝜎=5 and kernel size=5 and 10 (2 images)

    img_filtered_K5 = gaussian_smooth(img_Gray, kernel_size=5,sigma=SIGMA)
    img_filtered_K10 = gaussian_smooth(img_Gray, kernel_size=10,sigma=SIGMA)
    img_filtered_K5_normalized =  img_filtered_K5 / np.amax(img_filtered_K5) * 255 # normalized to 255 gray level
    img_filtered_K10_normalized =  img_filtered_K10 / np.amax(img_filtered_K10) * 255  # normalized to 255 gray level
    save_img_path1 = os.path.join('./results/Gaussian smooth results', 'gaussian_smooth_of_sigma_and_kernal_size_5.jpg')
    save_img_path2 = os.path.join('./results/Gaussian smooth results', 'gaussian_smooth_of_sigma_and_kernal_size_10.jpg')
    cv2.imwrite(save_img_path1, img_filtered_K5_normalized)
    cv2.imwrite(save_img_path2, img_filtered_K10_normalized)

    # ii. Sobel edge detection results
    # (1) magnitude of gradient (Gaussian kernel size=5 and 10) (2 images)
    # (2) direction of gradient (Gaussian kernel size=5 and 10) (2 images)    
    
    gradient_magnitude_K5, gradient_direction_K5 = sobel_edge_detection(img_filtered_K5_normalized)
    gradient_magnitude_K10, gradient_direction_K10 = sobel_edge_detection(img_filtered_K10_normalized)
    gradient_magnitude_K5_normalized =  gradient_magnitude_K5 / np.amax(gradient_magnitude_K5) * 255
    gradient_magnitude_K10_normalized =  gradient_magnitude_K10 / np.amax(gradient_magnitude_K10) * 255
    gradient_direction_K5_normalized =  gradient_direction_K5 / np.amax(gradient_direction_K5) * 255
    gradient_direction_K10_normalized =  gradient_direction_K10 / np.amax(gradient_direction_K10) * 255
    save_img_path3 = os.path.join('./results/Sobel edge detection results', 'magnitude_of_gradient_kernel_size_5.jpg')
    save_img_path4 = os.path.join('./results/Sobel edge detection results', 'magnitude_of_gradient_kernel_size_10.jpg')
    save_img_path5 = os.path.join('./results/Sobel edge detection results', 'direction_of_gradient_kernel_size_5.jpg')
    save_img_path6 = os.path.join('./results/Sobel edge detection results', 'direction_of_gradient_kernel_size_10.jpg')
    cv2.imwrite(save_img_path3, gradient_magnitude_K5_normalized)
    cv2.imwrite(save_img_path4, gradient_magnitude_K10_normalized)
    cv2.imwrite(save_img_path5, gradient_direction_K5_normalized)
    cv2.imwrite(save_img_path6, gradient_direction_K10_normalized)
    
    # iii. Structure tensor + NMS results (Gaussian kernel size=10)
    # (1) window size = 3x3 (1 image)
    # (2) window size = 30x30 (1 image)
    
    window_size=3
    response_matrix =structure_tensor(gradient_magnitude_K10, gradient_direction_K10, window_size, SIGMA, K)
    NMS_W3=NMS(response_matrix, THRESHOLD)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    plt.imshow(img_copy)

    is_corner_x = []
    is_corner_y = []

    for x in range(NMS_W3.shape[0]):
        for y in range(NMS_W3.shape[1]):
            if NMS_W3[x][y]:
                is_corner_x.append(x)
                is_corner_y.append(y)
    plt.plot(is_corner_y, is_corner_x, '+')
    plt.axis('off')
    plt.savefig("./results/Structure tensor + NMS results/NMS_window_size_3.jpg")
    
    window_size=30
    response_matrix =structure_tensor(gradient_magnitude_K10, gradient_direction_K10, window_size, SIGMA, K)
    NMS_W3=NMS(response_matrix, THRESHOLD)
    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    plt.imshow(img_copy)

    is_corner_x = []
    is_corner_y = []

    for x in range(NMS_W3.shape[0]):
        for y in range(NMS_W3.shape[1]):
            if NMS_W3[x][y]:
                is_corner_x.append(x)
                is_corner_y.append(y)
    plt.plot(is_corner_y, is_corner_x, '+')
    plt.axis('off')
    plt.savefig("./results/Structure tensor + NMS results/NMS_window_size_30.jpg")


    # B. Final results of rotating (by 30°) original images (1 image)    
        
    img_Gray_30 = rotate(img_Gray, ANGLE)
    img_filtered_K10_R30 = gaussian_smooth(img_Gray_30, kernel_size=10,sigma=SIGMA)
    gradient_magnitude_K10_R30, gradient_direction_K10_R30 = sobel_edge_detection(img_filtered_K10_R30)
    window_size=3
    response_matrix =structure_tensor(gradient_magnitude_K10_R30, gradient_direction_K10_R30, window_size, SIGMA, K)
    where_are_nan = np.isnan(response_matrix)
    response_matrix[where_are_nan] = 0
    NMS_W3=NMS(response_matrix, THRESHOLD)

    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    plt.imshow(img_Gray_30)

    is_corner_x = []
    is_corner_y = []

    for x in range(NMS_W3.shape[0]):
        for y in range(NMS_W3.shape[1]):
            if NMS_W3[x][y]:
                is_corner_x.append(x)
                is_corner_y.append(y)

    plt.plot(is_corner_y, is_corner_x, '+')
    plt.axis('off')
    plt.savefig("./results/Final results of rotating/Rotate_30.jpg")
    
    # C. Final results of scaling (to 0.5x) original images (1 image)
    
    img_Gray_scaled = cv2.resize(img_Gray,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    img_filtered_K10_scaled = gaussian_smooth(img_Gray_scaled, kernel_size=10,sigma=SIGMA)
    gradient_magnitude_K10_scaled, gradient_direction_K10_scaled = sobel_edge_detection(img_filtered_K10_scaled)
    window_size=3
    response_matrix =structure_tensor(gradient_magnitude_K10_scaled, gradient_direction_K10_scaled, window_size, SIGMA, K)
    where_are_nan = np.isnan(response_matrix)
    response_matrix[where_are_nan] = 0
    NMS_W3=NMS(response_matrix, THRESHOLD)

    plt.figure();plt.gray();plt.figure(figsize=(20,10))
    plt.imshow(img_Gray_scaled)

    is_corner_x = []
    is_corner_y = []

    for x in range(NMS_W3.shape[0]):
        for y in range(NMS_W3.shape[1]):
            if NMS_W3[x][y]:
                is_corner_x.append(x)
                is_corner_y.append(y)

    plt.plot(is_corner_y, is_corner_x, '+')
    plt.axis('off')
    plt.savefig("./results/Final results of scaling/Scaling.jpg")