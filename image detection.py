import cv2 as cv

frontal_face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

img = cv.imread('img.jpg')
grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = frontal_face_cascade.detectMultiScale(grayscale,1.3,5)


for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3) 
    roi_grayscale = grayscale[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_grayscale)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

while(1):
    cv.imshow('Image',img)
    k = cv.waitKey(0)
    if k==27:    # Press ESC key to stop
        cv.destroyAllWindows()
        break
    else:
        continue