import cv2
import numpy as np
import cvzone
from cvzone.ColorModule import ColorFinder

kf = cv2.KalmanFilter(4,2)
kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)


def estimate(x,y):
    measured = np.array([[np.float32(x)],[np.float32(y)]])
    kf.correct(measured)
    predicted = kf.predict()
    return predicted

def detectcontours(frame):
    x,y,w,h = 0,0,0,0
    contours,_=cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 500 :
            peri = cv2.arcLength(contour,True)
            approx= cv2.approxPolyDP(contour,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
    return x+w/2,y+h/2,w,h

def Masking(frame):
    lower= np.array([9,76,3])
    upper =np.array([94,255,255])
    mask = cv2.inRange(frame,lower,upper)
    return mask

    k = np.ones((10,10))
    maskdilated =cv2.dilate(mask,k)
    maskC= cv2.bitwise_and(frame,frame,mask=maskdilated)
    maskC= cv2.resize(maskC,(800,800))

    return maskdilated,maskC

def drawing(frame,points):
    for i in range(1,len(points)):
        cv2.line(frame,points[i-1],points[i],[0,0,255],2)

video=cv2.VideoCapture('green_ball.mp4')
points= []
predicted = np.zeros((2,1),np.float32)
count = 0

while True:
    success,frame = video.read()
    if success == True:
        mask= Masking(frame)
        maskC=Masking(frame)

        if cv2.waitKey(1) & 0XFF == ord('a'):
            count+=1
            cv2.imwrite("ball"+str(count)+".jpg",maskC)

        nextPoint=[]
        x,y,w,h = detectcontours(mask)
        nextPoint.append((int(x),int(y)))
        for i in nextPoint:
            points.append(i)

        predicted=estimate(x,y)
        if (0,0) in points:
            points.remove((0,0))
        if len(points)>15:
            del points[0]
        drawing(frame,points)

        cv2.circle(frame,(int(x),int(y)),20,[0,0,255],2,7)
        cv2.line(frame,(int(x),int(y+20)),(int(x+50),int(y)),[0,0,0],4,7)


        cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 20, [255, 0, 0], 2, 7)
        cv2.line(frame, (int(predicted[0])+16, int(predicted[1])-15), (int(predicted[0])+50, int(predicted[1])-30), [0, 0, 0], 4, 7)
        cv2.putText(frame, "Predicted", (int(predicted[0]+50), int(predicted[1]-30)), cv2.FONT_HERSHEY_PLAIN, 4, [255, 0, 0])
        frame=cv2.resize(frame,(800,800))
        cv2.imshow("final",frame)

        if cv2.waitKey(250) & 0XFF == ord("q"):
            break
    else:
        break


video.release()




