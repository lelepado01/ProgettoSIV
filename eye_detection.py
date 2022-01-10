import numpy as np
import cv2 
from math import sqrt


def getLeftAndRightEye(eyes): 
    (e1x, _, _, _) = eyes[0]
    (e2x, _, _, _) = eyes[1]

    left_eye = 1
    right_eye = 0
    if e1x < e2x: 
        left_eye = 0
        right_eye = 1

    return eyes[left_eye], eyes[right_eye] 


def getCenterOf(obj : tuple): 
    x, y, w, h = obj
    return x + w / 2, y + h / 2


def getFaceAngleFromEyes(left_eye, right_eye, face): 
    # IF LEFT > RIGHT -> LOOKING RIGHT (ratio > 1)
    # ELSE            -> LOOKING LEFT
    # 
    # Eye size determines global head direction (relative to camera)
    # Distance between eyes and face determine sight direction (?)

    face_center_x, face_center_y = getCenterOf(face)
    left_eye_center_x, left_eye_center_y = getCenterOf(left_eye)
    right_eye_center_x, right_eye_center_y = getCenterOf(right_eye)

    left_distance = sqrt((left_eye_center_x-face_center_x)**2 + (left_eye_center_y-face_center_y)**2)
    right_distance = sqrt((right_eye_center_x-face_center_x)**2 + (right_eye_center_y-face_center_y)**2)

    (_, _, lw, lh) = left_eye
    (_, _, rw, rh) = right_eye
    eye_ratio = lw * lh / (rw * rh) 
    
    global_sight_direction = 0
    if (eye_ratio > 1): 
        global_sight_direction = -1
    else: 
        global_sight_direction = 1

    relative_sight_direction = 0
    if left_distance > right_distance:  
        relative_sight_direction = 1
    else:
        relative_sight_direction = -1 

    return global_sight_direction, relative_sight_direction

def drawLookDirectionOnImage(img, left, right, face, global_angle, relative_angle): 
    (lx, ly, lw, lh) = left
    (rx, ry, rw, rh) = right
    (fx, fy, fw, fh) = face

    lx += fx
    ly += fy
    rx += fx
    ry += fy

    img = cv2.rectangle(img,(lx,ly),(lx+lw,ly+lh),(0,255,0),thickness=2)
    img = cv2.rectangle(img,(rx,ry),(rx+rw,ry+rh),(0,255,0),thickness=2)

    label = ""
    if global_angle > 0:
        label = "Looking right according to camera"
    else: 
        label = "Looking left according to camera"

    img = cv2.putText(img, label,(lx,ly-20),cv2.FONT_HERSHEY_SIMPLEX,1, (36,255,12), 2)

    if relative_angle > 0:
        label = "Looking right according to body"
    else: 
        label = "Looking left according to body"
    img = cv2.putText(img, label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,1, (36,255,12), 2)

    return img

#Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )


img = cv2.imread("imgs/rseba1.jpg");  

#Converting the recorded image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Applying filter to remove impurities
gray = cv2.bilateralFilter(gray,5,1,1)

faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(1,1))
for face in faces:
    (x,y,w,h) = face
    face_zoom = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_zoom,1.3,5,minSize=(1,1))

    if (len(eyes) == 2): 
        left_eye, right_eye = getLeftAndRightEye(eyes)
        global_angle, relative_angle = getFaceAngleFromEyes(left_eye, right_eye, face)
        img = drawLookDirectionOnImage(img, left_eye, right_eye, face, global_angle, relative_angle)

bodies, _ = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
if len(bodies) > 0: 
    for (x,y,w,h) in bodies:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(1,1))
if(len(faces)>0):
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
img = cv2.resize(img, (1080, 720))
cv2.imshow('img',img)
cv2.waitKey(0)

 
cv2.destroyAllWindows()