import cv2
import numpy as np

# load original image in grayscale
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# create salt-and-pepper noise
noise_prob = 0.05
noisy_image = image.copy()
num_salt = np.ceil(noise_prob * image.size * 0.5)
coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
noisy_image[coords[0], coords[1]] = 255

num_pepper = np.ceil(noise_prob * image.size * 0.5)
coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
noisy_image[coords[0], coords[1]] = 0

# apply median filter
median_filtered = cv2.medianBlur(noisy_image, 5)

cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow('Median Filtered Image', median_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
