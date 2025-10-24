import cv2

# load image in grayscale
image = cv2.imread('assets/images/lena.jpg', cv2.IMREAD_GRAYSCALE)

# apply simple thresholding
_, thresh_127 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # threshold = 127 if value is > 127 it will change to white
_, thresh_200 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # threshold = 200

# show results
cv2.imshow('Original Image', image)
cv2.imshow('Threshold 127', thresh_127)
cv2.imshow('Threshold 200', thresh_200)

cv2.waitKey(0)
cv2.destroyAllWindows()
