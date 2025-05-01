# processor.py
import cv2
import numpy as np

def apply_filter(image, operation: str):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if operation == "gaussian":
        noisy = add_gaussian_noise(gray)
        return cv2.GaussianBlur(noisy, (5, 5), 0)
    elif operation == "median":
        noisy = add_salt_pepper_noise(gray)
        return cv2.medianBlur(noisy, 5)
    elif operation == "bilateral":
        noisy = add_gaussian_noise(gray)
        return cv2.bilateralFilter(noisy, 9, 75, 75)
    elif operation == "canny":
        return cv2.Canny(gray, 100, 200)
    elif operation == "sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        return np.uint8(sobel)
    else:
        return gray

def add_gaussian_noise(image, mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def add_salt_pepper_noise(image, prob=0.05):
    output = np.copy(image)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
    return output
