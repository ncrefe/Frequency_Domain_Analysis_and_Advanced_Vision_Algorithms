# Seeing Images in the Frequency Domain, and Classic Vision Algorithms from Scratch

An image can be described by its pixels or by the *frequencies* it contains. Switching viewpoint makes some problems simple: blurring becomes multiplication, and the difference between structure and texture becomes visible. This project explores that, then rebuilds three classic algorithms (filtering, stereo matching and Canny edges) without relying on library implementations.

## 1. Parseval's theorem: no energy is lost

Transforming an image to the frequency domain and back does not create or destroy energy: the total energy in the pixels equals the total energy in the spectrum. The project proves this for the discrete Fourier transform, which is what makes the frequency view a faithful representation instead of an approximation.

## 2. Magnitude versus phase: which carries the picture?

The spectrum has two parts: **magnitude** (how much of each frequency) and **phase** (where it sits). Two images are transformed, and then *hybrid* images are built by combining the magnitude of one with the phase of the other. The result looks like the image that supplied the **phase**, not the magnitude. Phase carries the structure (edges, shapes, positions), while magnitude mostly describes overall texture and contrast. The difference between reconstructions is measured with the mean absolute difference.

## 3. Filtering: space or frequency, same answer

Box and Gaussian filters are implemented from scratch and applied in two ways: by sliding the kernel over the image (**spatial domain**), and by multiplying the spectra and transforming back (**frequency domain**, via the convolution theorem). The two results agree to numerical precision, confirming the theory. The frequency route becomes the cheaper one for large kernels.

## 4. Stereo matching with normalised cross-correlation

Given a left and right photo of the same scene, the horizontal shift of a patch between them (the **disparity**) tells you how far away it is. For every patch in the left image, NCC compares it against candidate patches along the same row in the right image. NCC is insensitive to brightness and contrast differences between the two cameras. The best match gives the disparity map, which is compared with OpenCV's block matcher using the mean absolute error.

## 5. Canny edge detection, step by step

Every stage is implemented: gradient computation, **non-maximum suppression** (thin edges to one pixel), **double thresholding** (strong, weak, discarded) and **edge tracking by hysteresis** (keep weak edges only if connected to strong ones). The output is compared with OpenCV's Canny using the mean absolute difference and the F1-score.

## What you can take from this

1. The Fourier transform is a lossless change of viewpoint, and phase matters more than magnitude for what a picture looks like.
2. The convolution theorem gives two equivalent ways to filter, and choosing between them is about cost.
3. Rebuilding an algorithm and checking it against a reference implementation is the best way to understand its details.

## Files

- `fourier_magnitude_phase.py`: spectra and hybrid image reconstruction
- `spatial_vs_frequency_filtering.py`: box and Gaussian filters in both domains
- `ncc_stereo_matching.py`: disparity map by normalised cross-correlation
- `canny_edge_detector.py`: the Canny pipeline from scratch
- `data/`: test images (including a stereo pair)
