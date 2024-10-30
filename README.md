# **Image Convolution and Plate Detection**
> *Scripts for image processing with convolution filters and vehicle plate detection using OCR.*

## **Introduction**
This project addresses the needs of image convolution filtering and automatic license plate detection. It provides tools to apply convolution filters, such as Gaussian and Sobel, to enhance or detect features within images and leverages OCR technology to extract plate numbers from images.

## **Project Description**
- **Main functionality:** Implements convolution on images using Gaussian (for noise reduction) and Sobel filters (for edge detection), and enables vehicle plate detection with OCR to read plate characters.
- **Technologies used:** Python, OpenCV, NumPy, Matplotlib, and Pytesseract.
- **Challenges faced:** Optimization of the convolution operation for different image types and precise plate contour detection.
- **Future improvements:** Adding support for more convolution filters and enhanced OCR accuracy with machine learning techniques.

## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Description](#project-description)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)

## **Installation**
1. **Prerequisites**:
   - **Python 3.x** - [Python Download](https://www.python.org/downloads/)
   - **OpenCV** - [OpenCV Documentation](https://opencv.org/)
   - **NumPy** - [NumPy Documentation](https://numpy.org/)
   - **Matplotlib** - [Matplotlib Documentation](https://matplotlib.org/)
   - **Pytesseract** - [Pytesseract Installation Guide](https://github.com/madmaze/pytesseract)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/ivmg5/Image-Convolution-And-Plate-Detection.git
   cd Image-Convolution-And-Plate-Detection
   ```

3. **Install dependencies**:
   ```bash
   pip install opencv-python-headless numpy matplotlib pytesseract
   ```

4. **Run the project**:
   To start, you can run either `convolution.py` for convolution filters or `plate_detection.py` for plate detection.

### **Configuration Options**
- Set `DEBUG=true` for debug information.
- Define environment variables such as `API_URL` if using network-dependent functionality in future integrations.

## **Usage**

### 1. Convolution on an Image (`convolution.py`)
Applies convolution filters on an input image. Available filters are Sobel (edge detection) and Gaussian (smoothing).

**Example usage**:
```bash
python convolution.py -i Turquia.jpg -f sobel -p same -k 3 -s 1.0
```

- **-i**: Path to the input image.
- **-f**: Filter to apply (options: `gaussian` or `sobel`).
- **-p**: Padding type (options: `valid` or `same`).
- **-k**: Kernel size (must be odd).
- **-s**: Sigma value for Gaussian kernel.

### 2. Plate Detection and OCR (`plate_detection.py`)
Detects the license plate in a vehicle image and extracts the characters using OCR.

**Example usage**:
```bash
python plate_detection.py -i Placa.jpg
```

- **-i**: Path to the input image.

## **License**
This project is licensed under the MIT License.

[![Build Status](https://img.shields.io/badge/status-active-brightgreen)](#)
[![Code Coverage](https://img.shields.io/badge/coverage-80%25-yellowgreen)](#)
