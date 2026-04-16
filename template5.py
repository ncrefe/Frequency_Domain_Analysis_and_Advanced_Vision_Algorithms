# Template for Exercise 5 – Canny Edge Detector

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def gaussian_smoothing(img, sigma):
    """
    Apply Gaussian smoothing to reduce noise.

    Arguments:
        img   : input grayscale image (numpy array)
        sigma : standard deviation of Gaussian kernel

    Returns:
        smoothed_img : Gaussian-smoothed image
    """
    # Determine kernel size: usually 6*sigma rounded up to next odd number
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    # Create 1D Gaussian kernel
    ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    gauss = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    gauss = gauss / np.sum(gauss)  # normalize

    # Create 2D Gaussian kernel by outer product
    kernel = np.outer(gauss, gauss)

    # Convolve image with Gaussian kernel
    smoothed_img = cv2.filter2D(img.astype(np.float32), -1, kernel)

    return smoothed_img


def compute_gradients(img):
    """
    Compute gradient magnitude and direction using Sobel filters.

    Arguments:
        img : input grayscale image (numpy array)

    Returns:
        gradient_magnitude : array of gradient magnitudes
        gradient_angle     : array of gradient directions in degrees [0,180)
    """

    # Sobel Operator:
    # The Sobel operator is used to approximate the first derivative of the image intensity.
    # It highlights regions with high spatial gradients (edges).
    #
    # Horizontal gradient (Gx):
    #   This kernel detects changes in intensity in the x-direction:
    #       [-1  0  1
    #        -2  0  2
    #        -1  0  1]
    #   Convolution with this kernel emphasizes vertical edges.
    #
    # Vertical gradient (Gy):
    #   This kernel detects changes in intensity in the y-direction:
    #       [-1 -2 -1
    #         0  0  0
    #         1  2  1]
    #   Convolution with this kernel emphasizes horizontal edges.
    #
    # The output Gx and Gy are the approximate derivatives along x and y directions, respectively.

    # Compute horizontal and vertical gradients
    Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Compute gradient magnitude
    gradient_magnitude = np.hypot(Gx, Gy)

    # Compute gradient angle (in degrees)
    gradient_angle = np.arctan2(Gy, Gx) * (180.0 / np.pi)
    gradient_angle[gradient_angle < 0] += 180  # Map angles to [0,180)

    return gradient_magnitude, gradient_angle


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.

    Arguments:
        mag : gradient magnitude (numpy array)
        ang : gradient angle in degrees [0,180) (numpy array)

    Returns:
        nms : thinned edges after non-maximum suppression
    """

    # Initialize output with zeros
    nms = np.zeros_like(mag, dtype=np.float32)
    h, w = mag.shape

    # Quantize angles to 4 main directions: 0, 45, 90, 135 degrees
    # This simplifies comparison with neighbors
    angle = ang.copy()
    angle[(angle >= 0) & (angle < 22.5)] = 0
    angle[(angle >= 22.5) & (angle < 67.5)] = 45
    angle[(angle >= 67.5) & (angle < 112.5)] = 90
    angle[(angle >= 112.5) & (angle < 157.5)] = 135
    angle[(angle >= 157.5) & (angle < 180)] = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            q = 255
            r = 255

            # Check neighbors along the gradient direction
            if angle[y, x] == 0:
                q = mag[y, x + 1]
                r = mag[y, x - 1]
            elif angle[y, x] == 45:
                q = mag[y - 1, x + 1]
                r = mag[y + 1, x - 1]
            elif angle[y, x] == 90:
                q = mag[y - 1, x]
                r = mag[y + 1, x]
            elif angle[y, x] == 135:
                q = mag[y - 1, x - 1]
                r = mag[y + 1, x + 1]

            # Suppress if not a local maximum
            if (mag[y, x] >= q) and (mag[y, x] >= r):
                nms[y, x] = mag[y, x]
            else:
                nms[y, x] = 0

    return nms


def double_threshold(nms, low, high):
    """
    Apply double thresholding to classify strong, weak, and non-edges.

    Arguments:
        nms : thinned edge map (after NMS)
        low : low threshold (float)
        high: high threshold (float)

    Returns:
        edge_map : array with 0=non-edge, 1=weak, 2=strong
    """

    strong = 255
    weak = 75
    edge_map = np.zeros_like(nms, dtype=np.uint8)

    # Classify pixels based on thresholds
    edge_map[nms >= high] = strong
    edge_map[(nms >= low) & (nms < high)] = weak

    return edge_map


def hysteresis(edge_map, weak=75, strong=255):
    """
    Perform edge tracking by hysteresis.

    Arguments:
        edge_map : edge map after double thresholding
        weak     : pixel value for weak edges
        strong   : pixel value for strong edges

    Returns:
        final_edges : binary edge map (0=non-edge, 255=edge)
    """
    # Copy the edge map to avoid modifying the original
    h, w = edge_map.shape
    final_edges = edge_map.copy()

    # Iterate over all pixels
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # If the pixel is weak, check 8 neighbors
            if final_edges[y, x] == weak:
                # If any neighbor is strong, promote to strong
                if ((final_edges[y - 1:y + 2, x - 1:x + 2] == strong).any()):
                    final_edges[y, x] = strong
                else:
                    # Otherwise, suppress
                    final_edges[y, x] = 0

    # Convert all strong pixels to 255 (binary edge)
    final_edges[final_edges == strong] = 255

    return final_edges


def compute_metrics(manual_edges, cv_edges):
    """
    Compute MAD, precision, recall, and F1-score between two binary edge maps.

    Arguments:
        manual_edges : binary edge map from your implementation (0 or 255)
        cv_edges     : binary edge map from cv2.Canny (0 or 255)

    Returns:
        metrics : dictionary with keys 'MAD', 'precision', 'recall', 'F1'
    """
    # Normalize to 0-1 for easier computation
    manual = (manual_edges > 0).astype(np.float32)
    cvmap = (cv_edges > 0).astype(np.float32)

    # Mean Absolute Difference (MAD)
    mad = np.mean(np.abs(manual - cvmap))

    # True positives: pixels detected as edge in both maps
    TP = np.sum((manual == 1) & (cvmap == 1))
    # False positives: pixels detected as edge in manual but not in cv_edges
    FP = np.sum((manual == 1) & (cvmap == 0))
    # False negatives: pixels detected as edge in cv_edges but not in manual
    FN = np.sum((manual == 0) & (cvmap == 1))

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP + 1e-10)  # avoid division by zero
    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN + 1e-10)
    # F1-score: harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {'MAD': mad, 'precision': precision, 'recall': recall, 'F1': f1}


# ==========================================================

# TODO: 1. Load the grayscale image 'bonn.jpg'
image = cv2.imread('data/bonn.jpg', cv2.IMREAD_GRAYSCALE)
# TODO: 2. Smooth the image using your Gaussian function
sigma = 0.3
smoothed = gaussian_smoothing(image, sigma)

# TODO: 3. Compute gradients (magnitude and direction)
grad_mag, grad_ang = compute_gradients(smoothed)

# TODO: 4. Apply non-maximum suppression
nms_edges = nonmax_suppression(grad_mag, grad_ang)

# TODO: 5. Apply double threshold (choose suitable low/high values)


low_thresh = 61.37   # optimal low threshold
high_thresh = 64.60  # optimal high threshold
dt_edges = double_threshold(nms_edges, low_thresh, high_thresh)

# TODO: 6. Perform hysteresis to obtain final edges
weak = 50
strong = 255
manual_edges = hysteresis(dt_edges, weak, strong)


# TODO: 7. Compare your result with cv2.Canny using MAD and F1-score
cv_edges = cv2.Canny(image, 50, 150)  # thresholds in OpenCV scale

metrics = compute_metrics(manual_edges, cv_edges)
print("MAD:", metrics['MAD'])
print("F1-score:", metrics['F1'])
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])

# TODO: 8. Display original image, your edges, and OpenCV edges
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(manual_edges, cmap='gray')
plt.title("Manual Canny Edges")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv_edges, cmap='gray')
plt.title("OpenCV Canny Edges")
plt.axis('off')

plt.tight_layout()
plt.show()

