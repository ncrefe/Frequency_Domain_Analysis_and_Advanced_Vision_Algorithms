# Template for Exercise 2 –  Fourier Transform and Image Reconstruction
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(img):
    """
    Compute the Fourier Transform of an image and return:
    - The shifted spectrum
    - The magnitude
    - The phase
    """
    freq_img = np.zeros_like((img), dtype=np.complex64)

    N,M = img.shape 
    x_grid = np.arange(M)
    y_grid = np.arange(N)

    X, Y = np.meshgrid(x_grid, y_grid)
    for u in range(N): 
        for v in range(M): 
            exp_matrix = np.exp(-2j*np.pi*((u * Y / N) + (v * X / M)))
            freq_img[u,v] = np.sum(exp_matrix*img)

    # Decompose in real and imaginary parts
    real_part = freq_img.real
    imaginary_part = freq_img.imag

    # Get Phase and Amplitude/Magnitude
    magnitude = np.sqrt(real_part**2 + imaginary_part**2)
    phase = np.arctan2(imaginary_part, real_part) 
    shift_spectrum = np.fft.fftshift(freq_img)

    return shift_spectrum, magnitude, phase


def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct an image from given magnitude and phase.
    """
    real_part = mag*np.cos(phase)
    img_part = mag*np.sin(phase) 

    complex_number = real_part+1j*img_part
    unshifted = np.fft.ifftshift(complex_number)

    image = np.fft.ifft2(unshifted)

    image_real_normed = np.clip(np.abs(image),0,255).astype(np.uint8)

    return image_real_normed


def compute_mad(a, b):
    """
    Compute the Mean Absolute Difference (MAD) between two images.
    """
    return np.mean(np.abs(a-b))

# ==========================================================

# TODO: 1. Load the two grayscale images (1.png and 2.png)
# TODO: 2. Compute magnitude and phase of both images
# TODO: 3. Swap magnitude and phase between the two images
# TODO: 4. Reconstruct and save the swapped results
# TODO: 5. Compute and print the MAD values between originals and reconstructions
# TODO: 6. Visualize all images (originals, magnitude, phase, reconstructions)

img1 = cv2.cvtColor(cv2.imread("data/1.png"), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread("data/2.png"), cv2.COLOR_BGR2GRAY)


print("\n=== Compute FT (will take a long time, due to manual implementation) ===")
shift_img1, magnitude_img1, phase_img1 = compute_fft(img1)
shift_img2, magnitude_img2, phase_img2 = compute_fft(img2)


# Swap magnitude and phase from the images. 
reconstruction_1 = reconstruct_from_mag_phase(magnitude_img1, phase_img2)
reconstruction_2 = reconstruct_from_mag_phase(magnitude_img2, phase_img1)


mad_original_recon1 = compute_mad(img1, reconstruction_1)
mad_original_recon2 = compute_mad(img2, reconstruction_2)
mad_cross_recon1 = compute_mad(img1, reconstruction_1)
mad_cross_recon2 = compute_mad(img2, reconstruction_2)

reconstruction_1 = reconstruct_from_mag_phase(magnitude_img1, phase_img2) # Mag1 + Phase2 
reconstruction_2 = reconstruct_from_mag_phase(magnitude_img2, phase_img1) # Mag2 + Phase1 

# MADs comparing Mag1 + Phase2
mad_1_vs_recon1 = compute_mad(img1, reconstruction_1)
mad_2_vs_recon1 = compute_mad(img2, reconstruction_1)

# MADs comparing Mag2 + Phase1
mad_1_vs_recon2 = compute_mad(img1, reconstruction_2)
mad_2_vs_recon2 = compute_mad(img2, reconstruction_2)


print("\n=== MAD Results ===")
print(f"MAD (Img1 vs. Recon_Mag1_Phase2): {mad_1_vs_recon1:.2f}") 
print(f"MAD (Img2 vs. Recon_Mag1_Phase2): {mad_2_vs_recon1:.2f}") 
print(f"MAD (Img1 vs. Recon_Mag2_Phase1): {mad_1_vs_recon2:.2f}") 
print(f"MAD (Img2 vs. Recon_Mag2_Phase1): {mad_2_vs_recon2:.2f}") 


# Visualize all the plots 
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1), plt.imshow(img1, cmap='gray'), plt.title('1. Original Img1')
plt.subplot(2, 4, 2), plt.imshow(img2, cmap='gray'), plt.title('2. Original Img2')
plt.subplot(2, 4, 3), plt.imshow(magnitude_img1, cmap='gray'), plt.title('3. Magnitude (from Img1)')
plt.subplot(2, 4, 4), plt.imshow(magnitude_img2, cmap='gray'), plt.title('4. Magnitude (from Img2)')

plt.subplot(2, 4, 5), plt.imshow(phase_img1, cmap='hsv'), plt.title('5. Phase (from Img1)')
plt.subplot(2, 4, 6), plt.imshow(phase_img2, cmap='hsv'), plt.title('6. Phase (from Img2)')
plt.subplot(2, 4, 7), plt.imshow(reconstruction_1, cmap='gray'), plt.title('7. Recon (Mag1 + Phase2)')
plt.subplot(2, 4, 8), plt.imshow(reconstruction_2, cmap='gray'), plt.title('8. Recon (Mag2 + Phase1)')

plt.tight_layout()
plt.show()



