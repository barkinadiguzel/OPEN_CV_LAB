import cv2

# read the image
image = cv2.imread('assets/images/lena.jpg')

# original image
cv2.imshow('1 - Original', image)

# resize to fixed size
resized_fixed = cv2.resize(image, (200, 200))
cv2.imshow('2 - Resized 200x200', resized_fixed)

# resize by scale factors
resized_scale = cv2.resize(image, None, fx=0.5, fy=0.5)
cv2.imshow('3 - Resized 0.5x0.5', resized_scale)

# keep aspect ratio while resizing
height, width = image.shape[:2]
scale_factor = 0.7
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
resized_aspect = cv2.resize(image, (new_width, new_height))
cv2.imshow('4 - Resized with aspect ratio', resized_aspect)

cv2.waitKey(0)
cv2.destroyAllWindows()
