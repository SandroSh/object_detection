import numpy as np
from scipy.ndimage import convolve
import cv2



# Canny algorithm consists
# Noise reduction;
# Gradient calculation;
# Non-maximum suppression;
# Double threshold;
# Hysteresis.

# This function represents a matrix to approximate the effect of applying a
# gaussian blur to an image.
# size determines size of a kernel in our case 5x5 matrix
# sigma controls the spread or width of the kernel. Larger sigma means that
# results will be more spread out.
def gaussian_kernel(size, sigma=1, image=None):
    # Generate the Gaussian kernel
    size = int(size) // 2
    # define a grid of coordinates ranging from -size to +size
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    # calculates the normalization factor  to ensure that the sum of the kernel values equals 1
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    # Ensure the sum of all kernel values is 1
    g /= g.sum()

    if image is not None:
        smoothed_image = convolve(image, g, mode='reflect')
        return smoothed_image

    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    # Calculates the magnitude of the gradient at each pixel
    G = np.hypot(Ix, Iy)
    # Normalizes the gradient values to the range 0–255 for visualization
    G = (G / G.max()) * 255
    theta = np.arctan2(Iy, Ix)

    #  return  a grayscale image showing edge intensities and a matrix with edge directions in radians
    return G, theta


# function eliminates pixels that are not local maxima in the gradient direction
def non_max_suppression(img, D):
    # D is gradient direction matrix in radians

    # create empty array with size of an image and convert raidans to degrees
    Z = np.zeros_like(img, dtype=np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    # process image's gradient to find strongest edges in specific direction
    for angle_min, angle_max, offset1, offset2 in [
        (0, 22.5, (0, 1), (0, -1)),
        (22.5, 67.5, (1, -1), (-1, 1)),
        (67.5, 112.5, (1, 0), (-1, 0)),
        (112.5, 157.5, (-1, -1), (1, 1))
    ]:
        mask = (angle >= angle_min) & (angle < angle_max)
        # shifts the image img by offset1[0] rows and offset1[1] columns
        shifted1 = np.roll(np.roll(img, offset1[0], axis=0), offset1[1], axis=1)
        shifted2 = np.roll(np.roll(img, offset2[0], axis=0), offset2[1], axis=1)

        # Ensure comparison results are boolean, and calculate Z correctly
        condition = (img[mask] >= shifted1[mask]) & (img[mask] >= shifted2[mask])
        Z[mask] = np.where(condition, img[mask], 0)

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    # gets the maximum pixel value in the image
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    strong = np.int32(255)
    weak = np.int32(25)

    # Create boolean masks that defines if pixel values are more or less than
    strong_mask = img >= highThreshold
    weak_mask = (img >= lowThreshold) & (img < highThreshold)

    # create new matrix and assign 255 and 25
    result = np.zeros_like(img, dtype=np.uint8)
    result[strong_mask] = strong
    result[weak_mask] = weak

    return result, weak, strong


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                if np.any(img[i-1:i+2, j-1:j+2] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny(image, low_threshold=0.05, high_threshold=0.15):
    # Gaussian smoothing with OpenCV (faster than SciPy)
    smoothed_image = gaussian_kernel(5, 1.4, image)

    # Sobel filters for gradient calculation
    gradient_magnitude, gradient_direction = sobel_filters(smoothed_image)

    # Non-maximum suppression
    thin_edges = non_max_suppression(gradient_magnitude, gradient_direction)

    # Double threshold
    thresholded_image, weak, strong = threshold(thin_edges, lowThresholdRatio=low_threshold,
                                                highThresholdRatio=high_threshold)
    # Hysteresis
    edges = hysteresis(thresholded_image, weak, strong)

    return edges