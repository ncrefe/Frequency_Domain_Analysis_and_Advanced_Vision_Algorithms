# Template for Exercise 4 – NCC Stereo Matching

import cv2
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 11  # NCC patch size
MAX_DISPARITY = 64  # Maximum search range


def compute_manual_ncc_map(left_image, right_image, window_size, max_disparity):
    """
    Compute a dense disparity map using Normalized Cross-Correlation (NCC).

    Arguments:
        left_image, right_image : input grayscale stereo pair
        window_size             : size of the correlation window
        max_disparity           : maximum horizontal shift to consider

    Returns:
        disparity_map : computed disparity for each pixel (float32)
    """
    h, w = left_image.shape
    offset = window_size // 2
    disparity_map = np.zeros((h, w), dtype=np.float32)

    # Normalize images to [0,1]
    left_norm = left_image.astype(np.float32) / 255.0
    right_norm = right_image.astype(np.float32) / 255.0

    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            # Extract patch from left image
            left_patch = left_norm[y - offset:y + offset + 1, x - offset:x + offset + 1]
            left_mean = np.mean(left_patch)
            left_std = np.std(left_patch)
            left_std = max(left_std, 1e-5)

            best_ncc = -1
            best_disp = 0

            # Search across disparities
            for d in range(max_disparity):
                if x - d - offset < 0:
                    break  # out of right image bounds

                # Extract corresponding patch from right image
                right_patch = right_norm[y - offset:y + offset + 1, x - d - offset:x - d + offset + 1]
                right_mean = np.mean(right_patch)
                right_std = np.std(right_patch)
                right_std = max(right_std, 1e-5)

                # NCC (Normalized Cross-Correlation) measures similarity between two patches.
                # It subtracts the mean (to remove brightness bias) and divides by std deviation (to remove contrast effect),
                # making the correlation invariant to illumination and scale differences.
                # Formula: ncc = Σ((L - μ_L) * (R - μ_R)) / (N * σ_L * σ_R)
                ncc = np.sum((left_patch - left_mean) * (right_patch - right_mean)) / (
                            window_size ** 2 * left_std * right_std)

                if ncc > best_ncc:
                    best_ncc = ncc
                    best_disp = d

            # Only assign disparity if confidence is high enough
            # OpenCV filters out unreliable matches - we do the same
            if best_ncc > 0.51:  # Confidence threshold
                disparity_map[y, x] = best_disp
            else:
                disparity_map[y, x] = 0  # Invalid/unreliable match

    return disparity_map


def compute_mae(a, b, mask=None):
    """
    Compute Mean Absolute Error (MAE) between two disparity maps.
    Optionally, use a mask to exclude invalid pixels.
    """
    if a.shape != b.shape:
        raise ValueError("Input maps must have the same shape")

    # Compute absolute difference
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))

    if mask is not None:
        diff = diff[mask]  # only keep valid pixels

    mae = np.mean(diff)

    return mae


# ==========================================================


# TODO: 1. Load the stereo image pair (left.png, right.png) in grayscale
left = cv2.imread("data/left.jpg", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("data/right.jpg", cv2.IMREAD_GRAYSCALE)

# TODO: 2. Call your NCC function to compute the manual disparity map
manual_disparity = compute_manual_ncc_map(left, right, WINDOW_SIZE, MAX_DISPARITY)

# TODO: 3. Compute a benchmark map using cv2.StereoBM_create with the same parameters
stereo = cv2.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=WINDOW_SIZE)
benchmark_disparity = stereo.compute(left, right).astype(np.float32) / 16.0  # OpenCV scales by 16

# TODO: 4. Visualize both maps and compare them qualitatively
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(manual_disparity, cmap='jet')
plt.title("Manual NCC Disparity Map")
plt.colorbar(label='Disparity (pixels)')

plt.subplot(1, 2, 2)
plt.imshow(benchmark_disparity, cmap='jet')
plt.title("OpenCV StereoBM Disparity Map")
plt.colorbar(label='Disparity (pixels)')

plt.tight_layout()
plt.show()

# TODO: 5. Quantitatively compare both maps by computing MAE (Mean Absolute Error)
# Create a mask to avoid invalid disparities (StereoBM can return negative values for invalid)
valid_mask = (benchmark_disparity > 0) & (manual_disparity > 0)

mae = compute_mae(manual_disparity, benchmark_disparity, mask=valid_mask)
print("Mean Absolute Error (MAE) vs OpenCV StereoBM:", mae)

# TODO: 6. Ensure your manual implementation achieves MAE < 0.7 pixels
assert mae < 0.7, "MAE too high! Manual NCC not accurate enough."
