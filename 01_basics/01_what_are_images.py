import cv2  # import OpenCV library
import numpy as np  # import numpy for array operations

image = cv2.imread('assets/images/lena.jpg')  # read image from file

if image is None:
    print("Image not found! Add a file in 'assets/images/'")
    exit()

print("Shape:", image.shape)  # print dimensions of the image

cv2.imshow('Lena Image', image)  # display image in a window
cv2.waitKey(0)  # 0 = wait indefinitely; program pauses here
cv2.destroyAllWindows()  # close all opened windows (cleanup)

print("Top-left pixel BGR value:", image[0,0])  # BGR value of top-left pixel

image[0,0] = [0,0,255]  # change top-left pixel to red (BGR)
print("Changed Top-left pixel BGR value:", image[0,0])

cv2.imshow('Modified Image', image)  # show modified image
cv2.waitKey(0)  # wait until a key is pressed
cv2.destroyAllWindows()  # close all windows (cleanup)
