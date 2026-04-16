# 📷 Sheet 02 — Frequency Domain Analysis & Advanced Vision Algorithms

This project focuses on advanced computer vision techniques involving **Fourier analysis, spatial vs frequency filtering, stereo matching, and edge detection**. It emphasizes both theoretical understanding and practical implementation of core algorithms from scratch.

The goal is to explore how image information is represented and processed in different domains, and to evaluate the performance of custom implementations against standard methods.

---

## 🚀 Features

* 🧠 Theoretical analysis of Parseval’s Theorem
* 🌊 Fourier Transform visualization (magnitude & phase)
* 🔄 Image reconstruction using swapped frequency components
* 🧹 Spatial vs frequency domain filtering
* 📊 Quantitative evaluation using MAD and MAE
* 👁️ Stereo matching using Normalized Cross-Correlation (NCC)
* 🗺️ Disparity map generation and comparison
* ✂️ Custom implementation of Canny Edge Detector
* 📈 Performance evaluation using F1-score

---

## 🛠️ Tech Stack

* Python 3.12
* NumPy 2.3.3
* OpenCV 4.11
* matplotlib

> ⚠️ Designed for Linux-based environments
> ⚠️ Only allowed libraries are used (NumPy, OpenCV, matplotlib)

---

## 📂 Project Structure

```
.
├── 1.png
├── 2.png
├── lena.png
├── bonn.jpg
├── left.png
├── right.png
├── q2_fourier.py
├── q3_filtering.py
├── q4_ncc.py
├── q5_canny.py
├── README.md
```

---

## ⚙️ Installation

```
pip install opencv-python numpy matplotlib
```

---

## ▶️ Usage

Run each module independently:

```
python q2_fourier.py
python q3_filtering.py
python q4_ncc.py
python q5_canny.py
```

---

## 🧪 Implemented Tasks

### 1. Parseval’s Theorem (Theory)

* Proof of energy preservation between spatial and frequency domains
* Verification using discrete Fourier transform formulation

---

### 2. Fourier Transform & Image Reconstruction

* Compute and visualize:

  * Magnitude spectrum
  * Phase spectrum

* Reconstruct hybrid images:

  * Magnitude (Image 1) + Phase (Image 2)
  * Magnitude (Image 2) + Phase (Image 1)

* Evaluate differences using Mean Absolute Difference (MAD)

---

### 3. Filtering in Spatial & Frequency Domains

* Implement from scratch:

  * Box filter
  * Gaussian filter

* Apply filters:

  * In spatial domain
  * In frequency domain (via FFT)

#### 📊 Evaluation

* Compare outputs visually
* Compute Mean Absolute Difference (MAD)
* Ensure numerical consistency between domains

---

### 4. Normalized Cross-Correlation (NCC)

* Implement NCC from scratch for stereo matching
* Generate disparity map manually

#### 🔍 Benchmark Comparison

* Compare with OpenCV StereoBM implementation
* Compute Mean Absolute Error (MAE)
* Ensure close agreement with reference results

---

### 5. Canny Edge Detection

* Implement full Canny pipeline from scratch:

  * Gradient computation
  * Non-maximum suppression
  * Double thresholding
  * Edge tracking

* Compare with OpenCV implementation

#### 📈 Evaluation

* Compute:

  * Mean Absolute Difference (MAD)
  * F1-score

---

## 📌 Notes

* All algorithms are implemented with minimal reliance on built-in functions
* Focus is on understanding underlying principles rather than using abstractions
* Quantitative evaluation is used to validate correctness and performance
* This project extends concepts from spatial image processing to frequency analysis
