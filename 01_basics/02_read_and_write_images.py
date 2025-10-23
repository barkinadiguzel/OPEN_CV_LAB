import cv2  

# read an image
image = cv2.imread('assets/images/lena.jpg')  # load image from file

if image is None:
    print("Image not found! Add a file in 'assets/images/'")
    exit()

# show the image
cv2.imshow('Original Image', image)  # display image in a window
cv2.waitKey(0)  # wait until a key is pressed
cv2.destroyAllWindows()  # close all windows

# save the image
cv2.imwrite('assets/images/lena_copy.jpg', image)  # save image as JPEG
cv2.imwrite('assets/images/lena_copy.png', image)  # save image as PNG

# read the saved PNG image
png_image = cv2.imread('assets/images/lena_copy.png')  # load PNG file
cv2.imshow('PNG Image', png_image)  # display the PNG image
cv2.waitKey(0)  # wait until a key is pressed
cv2.destroyAllWindows()  # close all windows 
