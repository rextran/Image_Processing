#import libraries of python opencv
import cv2
import numpy as np

#create VideoCapture object and read from video file
cap = cv2.VideoCapture('carMonitoring.mp4')
# use trained cars XML classifiers
car_cascade = cv2.CascadeClassifier('car2.xml')

#read until video is completed
while True:
    #capture frame by frame
    ret, frame = cap.read()
    #convert video into gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect cars in the video
    cars = car_cascade.detectMultiScale(gray, scaleFactor=5, minNeighbors=3, minSize=(20, 20))

    #to draw arectangle in each cars
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x,y,w,h) in cars:
        number = 0
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,"car",(x,y+h), font, 0.5,(0,255,255), 1, cv2.LINE_AA)
        number += 1
    #display the resulting frame
    cv2.imshow('video', frame)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == 27:
        break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()
