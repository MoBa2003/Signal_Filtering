# Digital Signal and Image Processing Projects

This repository contains three mini-projects focused on signal and image processing using digital filters, Fourier transforms, and convolution operations. Each project explores a different aspect of real-world signal and image analysis.

---

## Project 1: **Analysis and Design of Digital Filters for Audio Signal Processing**

### 📌 Objective:
To analyze and process real-world audio signals (e.g., human speech or music) using digital filters. The goal is to use Fourier Transform for frequency analysis and apply low-pass and high-pass filters to remove noise and isolate desired frequency bands.

### 🔧 Key Components:
- Load and digitize an audio signal.
- Perform frequency analysis using Fast Fourier Transform (FFT).
- Implement digital filters:
  - **Low-pass filter**: to retain low-frequency components (e.g., bass in audio).
  - **High-pass filter**: to retain high-frequency components (e.g., treble or sharp sounds).
- Visualize both time-domain and frequency-domain results before and after filtering.

---

## Project 2: **Convolution of Random Signals and Frequency Analysis**

### 📌 Objective:
To apply convolution operations to random and nonlinear signals and analyze how convolution affects the signal in the frequency domain.

### 🔧 Key Components:
- Generate or import random and nonlinear signals.
- Apply convolution between signals.
- Analyze changes in signal amplitude and frequency characteristics after convolution using:
  - Time-domain visualization.
  - Frequency-domain analysis (e.g., FFT).

---

## Project 3: **Edge Detection in Images Using Convolution**

### 📌 Objective:
To perform edge detection and feature extraction in an image using custom convolution filters instead of traditional ones (e.g., Sobel, Gaussian).

### 🧭 Steps:
1. **Load and display the image**.
2. **Create a custom convolution kernel** for detecting specific edges or patterns.
3. **Apply the convolution operation** to the image.
4. **Visualize the result**, highlighting detected edges or features.

### 🔧 Tools and Libraries Used:
- Python
- NumPy
- SciPy
- Matplotlib
- OpenCV (optional for image handling)
- Librosa or Scipy.io.wavfile (for audio loading)

---

## 🔄 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/signal-image-processing-projects.git
   cd signal-image-processing-projects
