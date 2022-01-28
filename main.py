import cv2
import time

# choose you're camera it can be 0 or 1 or higher number
capture = cv2.VideoCapture(0)
# definition of size window
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)
faceCasscade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


while True:
    # capture video from camera
    _, frame = capture.read()
    # convert to gray
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find face
    faces = faceCasscade.detectMultiScale(
        grayscale,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50,50)
    )

    for x, y, face_width, face_height in faces:
        # rectangle face
        #cv2.rectangle(grayscale, (x, y), (x+face_width, y + face_height), (255,0,0), 5)
        # finding face to blurring
        blur = cv2.blur(frame[y:y+face_height, x:x+face_width], ksize=(50,50))
        # replace part of image blured
        frame[y:y + face_height, x:x + face_width] = blur

    # show image in window
    cv2.imshow('frame', frame)
    # to show grayscale window
    #cv2.imshow('gray', grayscale)
    # wait for escape key then stop action
    key = cv2.waitKey(50)
    if key == 27:
        break

capture.release()
