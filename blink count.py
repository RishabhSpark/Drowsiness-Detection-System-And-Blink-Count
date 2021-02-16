from pygame import mixer                        # audio
import cv2 as cv                                # video 
import dlib                                     # facial landmark detection
import scipy.spatial
from imutils import face_utils      
import numpy as np             

eye_aspect_ratio_threshold = 0.21                 # eye aspect ratio threshold
eye_aspect_ratio_consecutive_frames = 1           # consecutive frames for which the code has detected closed eyes
count = 0
blink = 0

face_cascade = cv.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")         # haar cascade of face detection
detector = dlib.get_frontal_face_detector()                                                     
predictor = dlib.shape_predictor('shape predictor\shape_predictor_68_face_landmarks.dat')       # shape predictor

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    left_right = scipy.spatial.distance.euclidean(eye[0], eye[3])                 #distance between left to right
    top_left = scipy.spatial.distance.euclidean(eye[1], eye[5])                   #distance between point top left to bottom left
    top_right = scipy.spatial.distance.euclidean(eye[2], eye[4])                  #distance between point top right to bottom right
    
    eyes_ratio = (top_left+top_right)/(2*left_right)
    return eyes_ratio


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']                    # facial landmark indexes for left and right eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

video_capture = cv.VideoCapture(0)                                               # video capture
video_capture.set(cv.CAP_PROP_FPS, 30)                                           # fps set to 30

while(True):
    # Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv.flip(frame,1)
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect facial points through detector function
    faces = detector(grayscale, 0)

    face_rectangle = face_cascade.detectMultiScale(grayscale, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    #Detect facial points
    for face in faces:

        shape = predictor(grayscale, face)
        shape = face_utils.shape_to_np(shape)

        # extracting the left and right eye from indexes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]


        # coordinates to compute the eye aspect ratio for both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        EyeAspectRatio_Average = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        # convex hull around both of the eyes
        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)
        cv.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)              # drawing curves around left eye 
        cv.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)             # drawing curves around right eye

        if  EyeAspectRatio_Average < eye_aspect_ratio_threshold:
                count += 1
        else:
            if count >= eye_aspect_ratio_consecutive_frames:
                blink += 1
            count = 0
            cv.putText(frame, "Blinks: {}".format(blink), (20, 40), cv.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)

    cv.imshow('Video', frame)
    k = cv.waitKey(1)
    if (k==27):                        #ESC Key to exit
        break

video_capture.release()
cv.destroyAllWindows()