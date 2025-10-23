import cv2  

# open webcam (0) or a video file ('video.mp4')
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Cannot open video")
    exit()

# setup video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # codec
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))  # filename, codec, fps, frame size

while True:
    ret, frame = cap.read()  # read a frame
    if not ret:
        break  # stop if no frame is returned

    frame_resized = cv2.resize(frame, (640,480))  # resize frame to match writer size
    out.write(frame_resized)  # write the frame to video file

    cv2.imshow('Video Frame', frame_resized)  # display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # wait 1 ms, exit if 'q' pressed
        break

cap.release()  # release webcam/video
out.release()  # release video writer
cv2.destroyAllWindows()  # close all windows (cleanup)
