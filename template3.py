# Template for Exercise 3 – Spatial and Frequency Domain Filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    return 1/(k*k) * np.ones((k,k), dtype=np.float64)

def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    # Mehsgird around 0,0 center
    center = k // 2
    x_axis = np.arange(k) - center
    y_axis = np.arange(k) - center 

    X,Y = np.meshgrid(x_axis, y_axis)
    gauss_factor = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) # Normalize with something else ?
    return gauss_factor / np.sum(gauss_factor)



def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    # Make zero padding 
    K = h.shape[0]
    P = K // 2

    H, W = img.shape
    padded_img = np.zeros((H+2*P, W+2*P), dtype=np.float64)
    padded_img[P:H+P,P:W+P] = img 

    output = np.zeros_like(img, dtype=np.float64) 

    # Simple sliding window conv, which only works for Symmetric Kernels (else we would have to flip)
    for i in range(H): 
        for j in range(W): 
            kernel_region = padded_img[i:i+K,j:j+K]
            output[i,j] = np.sum(kernel_region*h)

    return output


def freq_linear_conv(img, h):
    """
    Perform linear convolution in the frequency domain.
    (You can use numpy.fft)
    """
    H, W = img.shape
    K,_ = h.shape
    n_min = H+K-1

    N = 2**int(np.ceil(np.log2(n_min)))
    M = 2**int(np.ceil(np.log2(n_min))) 
    
    img_padded = np.zeros((N, M), dtype=np.float64)
    h_padded = np.zeros((N, M), dtype=np.float64)

    img_padded[:H,:W] = img.astype(np.float64) 
    h_padded[:K,:K] = h.astype(np.float64)

    IMG_F = np.fft.fft2(img_padded)
    H_F = np.fft.fft2(h_padded)

    CONV_F = IMG_F*H_F

    output_padded = np.fft.ifft2(CONV_F)

    padding_length = K // 2
    output = np.real(output_padded)[padding_length:H+padding_length, padding_length:W+padding_length]

    return output

def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    return np.mean(np.abs(a-b))

# ==========================================================

# TODO: 1. Load the grayscale image (e.g., lena.png)
# TODO: 2. Construct 9×9 box and Gaussian kernels (same sigma)
# TODO: 3. Apply both filters spatially (manual convolution)
# TODO: 4. Apply both filters in the frequency domain
# TODO: 5. Compute and print MAD between spatial and frequency outputs
# TODO: 6. Visualize all results (original, box/gaussian spatial, box/gaussian frequency, spectrum)
# TODO: 7. Verify that MAD < 1×10⁻⁷ for both filters

# Load the image (GRAYSCALE, i assume)
img = cv2.cvtColor(cv2.imread("data/lena.png"), cv2.COLOR_BGR2GRAY).astype(np.float64)

# 1.5 sigma by taking rule of thumb that half-widht of the kernel should roughly be 3*sigma
SIGMA = 1.5
KERNEL_SIZE = 9
# Apply the filters on the image 
print("Apply the kernels in the spatial domain...")
spatial_gaussian = conv2_same_zero(img, make_gauss_kernel(KERNEL_SIZE, SIGMA))
spatial_box = conv2_same_zero(img, make_box_kernel(KERNEL_SIZE))

print("Apply the kernels in the frequency domain...")
fft_gaussian = freq_linear_conv(img , make_gauss_kernel(KERNEL_SIZE, SIGMA))
fft_box = freq_linear_conv(img, make_box_kernel(KERNEL_SIZE))

print("Computing MAD between the spatial and frequency based filtered solutions...")
mad_gaussian = compute_mad(spatial_gaussian, fft_gaussian)
mad_box = compute_mad(spatial_box, fft_box)

print("\n=== Mean Absolute Difference (MAD) ===")
print(f"Box Filter MAD (Spatial vs. Freq): {mad_box:.10f}")
print(f"Gaussian Filter MAD (Spatial vs. Freq): {mad_gaussian:.10f}")

# Show that we are within the tolerance of the assignment 
EPS = 10e-7 
print("\n === Within the tolerance? ===  ")
print(f"Gaussian filter within tolerance: {mad_gaussian < EPS}")
print(f"Box filter within tolerance {mad_box < EPS}")



plt.figure(figsize=(15, 10))

titles = [
    "1. Original Image", 
    "2. Box Filter (Spatial)", 
    "3. Gaussian Filter (Spatial)", 
    "4. Box Filter (Frequency)", 
    "5. Gaussian Filter (Frequency)",
]

images = [
    img, 
    spatial_box, 
    spatial_gaussian, 
    fft_box, 
    fft_gaussian, 
]

for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 3, i + 1)
    # Use 'gray' colormap and clip/scale results for display (0-255)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255) 
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()