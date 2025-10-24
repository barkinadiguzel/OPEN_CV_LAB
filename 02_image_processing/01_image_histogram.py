# Histogram: A graph showing the distribution of pixel intensity values in an image
# Mainly used to analyze brightness and contrast levels

import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image in grayscale mode
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# calculate histogram using OpenCV
hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # 256 intensity bins

# display the image
cv2.imshow('Grayscale Image', image)

# plot the histogram
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
