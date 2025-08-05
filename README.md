# Motion Detection with Canny Edge Detection

This project processes a video to detect motion using frame differencing and edge detection. It includes a full custom implementation of the Canny edge detector and compares it with OpenCV's built-in method.

## What it does

The script performs the following steps:

1. Reads a video file frame by frame.
2. Converts each frame to grayscale.
3. Calculates the absolute difference between the current and previous frame to highlight motion.
4. Applies edge detection using either a custom Canny function or OpenCV's `cv2.Canny`.
5. Finds contours in the edge map and draws bounding boxes around moving objects.

The final output is a real-time display showing both the motion mask and the video frame with detected motion highlighted.

## Files

- `Canny.py`: Custom implementation of the Canny edge detection algorithm (Gaussian blur, gradient, non-maximum suppression, thresholding, and hysteresis).
- `main.py`: The main script for video processing and motion detection.
- `bData2.mp4`: The video input file. You need to supply this or change the path in the script.

## How to run

First, install the required libraries:

```bash
pip install numpy opencv-python scipy

python main.py
