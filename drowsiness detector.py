from pygame import mixer                        # audio
import cv2 as cv                                # video 
import dlib                                     # facial landmark detection
import scipy.spatial
from imutils import face_utils      
import numpy as np             

mixer.init()                                   # pygame mixer initialize
mixer.music.load('alarm.WAV')                  # audio load

eye_aspect_ratio_threshold = 0.25                   # eye aspect ratio threshold
eye_aspect_ratio_consecutive_frames = 60            # consecutive frames for which the code has detected closed eyes 
                                                    #(score which  will be compared to count)
count = 0

face_cascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')         # haar cascade of face detection
detector = dlib.get_frontal_face_detector()                                                     
predictor = dlib.shape_predictor('shape predictor\shape_predictor_68_face_landmarks.dat')       # shape predictor

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    left_right = scipy.spatial.distance.euclidean(eye[0], eye[3])                 #displacement between left to right
    top_left = scipy.spatial.distance.euclidean(eye[1], eye[5])                   #displacement between point top left to bottom left
    top_right = scipy.spatial.distance.euclidean(eye[2], eye[4])                  #displacement between point top right to bottom right
    
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
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),3)

    # Detect facial points
    # We extract the indexes of the given facial part and grab the (x, y)-coordinates from the shape NumPy array
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
        EyeAspectRatio_Average_Round = round(EyeAspectRatio_Average, 2)


        # convex hull around both of the eyes
        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)
        cv.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)              # drawing curves around left eye
        cv.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)             # drawing curves around right eye

        x = 10
        y = 40

        cv.putText(frame, "Eyes Aspect Ratio: {}".format(EyeAspectRatio_Average_Round), (x, y), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

        # If eye aspect ratio is less than threshold i.e 0.25 then count is increased by 1
        if EyeAspectRatio_Average < eye_aspect_ratio_threshold:
            count += 1
            #If no. of frames is greater than consecutive frames i.e 60 then alarm will be played 
            if count >= eye_aspect_ratio_consecutive_frames:
                mixer.music.set_volume(0.3)
                mixer.music.play(-1)

                cv.putText(frame, "DROWSINESS ALERT!!!", (100,200), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                cv.putText(frame, "    Hopefully ???", (200,250), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1)
        else:
            mixer.music.stop()
            count = 0


    cv.imshow('Video', frame)
    k = cv.waitKey(1)
    if (k==27):                        #ESC Key to exit
        break

video_capture.release()
cv.destroyAllWindows()