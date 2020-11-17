import cv2
import pafy
import time 
count = 0

while(True):
    if count is 4:
        break
    count = count + 1
    print('IA procurando ameacas...')
    time.sleep(2)

#Using haarcascade_frontalface, detect faces from videos
# Source reference link:
# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
"""
url = 'www.youtube.com/watch?v=aKX8uaoy9c8'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")
videoSource = play.url
"""
"""
videoSource = "videos/v3.mp4"
cap = cv2.VideoCapture(videoSource)

# Load the cascade
face_cascade = cv2.CascadeClassifier('modelos/haarcascade_frontalface/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('modelos/haarcascade_eye/haarcascade_eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier('modelos/haarcascade_eye_tree_eyeglasses/haarcascade_eye_tree_eyeglasses.xml')


while(True):
    if not cap.isOpened():
        cap.open()

    # Capture frame-by-frame
    ret, img = cap.read() # frame is the image fram from the video and ret is the status of the frame (true for ok and false for error)
    if ret:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
#            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray= gray[y:y+h, x:x+w]
            roi_color= img[y:y+h, x:x+w]
            eyes = eyes_cascade.detectMultiScale(roi_gray)
#           print(eyes)
#           for (ex,ey,ew,eh) in eyes:
#               cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyeglasses = eyeglasses_cascade.detectMultiScale(roi_gray)
#            print(eyeglasses)
            if(len(eyes) or len(eyeglasses)):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#           for (egx,egy,egw,egh) in eyeglasses:
#               cv2.rectangle(roi_color,(egx,egy),(egx+egw,egy+egh),(255,0,0),2)
        # Display
        cv2.imshow('img', img)

        k = cv2.waitKey(40)
        
        if k == ord('q'):
            break
    else:
        print("Video Stopped")
        break

cap.release()
"""

###########################################################################################################################
###########################################################################################################################
#Using haarcascade_frontalface, detect faces from images
# Source reference link:
# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81

# Load the cascade
face_cascade = cv2.CascadeClassifier('modelos/haarcascade_frontalface/haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('images/img3.jpeg')

# Convert into grayscale
#gray = cv2.imread('images/test.jpg', 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) #BGR
    cv2.putText(img, 'Ameaca Detectada', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

"""
General source reference link:

https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://answers.opencv.org/question/71381/how-to-use-opencv-to-find-if-a-person-is-wearing-a-hat-or-glasses/
https://www.youtube.com/watch?v=88HdqNDQsEk
https://docs.opencv.org/3.4/db/d7c/group__face.html
https://docs.opencv.org/3.4/dc/dd7/classcv_1_1face_1_1BasicFaceRecognizer.html
https://docs.opencv.org/3.4/dd/d65/classcv_1_1face_1_1FaceRecognizer.html
https://towardsdatascience.com/facial-keypoints-detection-deep-learning-737547f73515
https://docs.opencv.org/3.4/d5/df6/group__dnn__objdetect.html
https://github.com/opencv/opencv/tree/master/data/haarcascades
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
https://www.geeks3d.com/hacklab/20200310/python-3-and-opencv-part-4-face-detection-webcam-or-static-image/
"""
