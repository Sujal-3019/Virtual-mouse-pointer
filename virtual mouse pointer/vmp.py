import cv2
import sujal_hand_track_module as stml
import time
import numpy as np
import pyautogui as pa

wcam , hcam=640,480
framer=100 #frame reduction
smooth=1

wscr,hscr=pa.size()
# print(wscr,hscr)

cap=cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

ptime =0
plocx ,plocy=0,0
clocx ,clocy=0,0


detector=stml.handDetector(maxHands=1)

while True:
    # find hand landmarks
    succ , img =cap.read()
    img=cv2.flip(img,1)
    img=detector.findHands(img)
    lmlist , bbox=detector.findposition(img)
    
    # get the tip of index and middle finger
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        # check which finger is up
        fingers=detector.fingersup()
        # print(fingers)
        # if only index finger is up:moving mode
        cv2.rectangle(img,(framer,framer),(wcam-framer,hcam-framer),(0,255,255),2) #if hand is detected then a recatangle is created
        if fingers[1]==1 and fingers[2]==0:

            # convert coodinates
            x3=np.interp(x1,(framer,wcam-framer),(0,wscr)) #if dont want the boundation of the box to work as full screen then just replace the framer with 0
            y3=np.interp(y1,(framer,hcam-framer),(0,hscr)) #if dont want the boundation of the box to work as full screen then just replace the framer with 0

            # smooth values
            clocx=plocx + (x3-plocx) / smooth
            clocy=plocy + (y3-plocy) / smooth

            # moving mouse
            pa.moveTo(clocx,clocy)
            cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
            plocx,plocy=clocx,clocy

        # if both (index and middle) fingers are up the : click mode
        if fingers[1]==1 and fingers[2]==1:
            # finding distance between fingers
            length,img,lineinfo =detector.findDistance(8,12,img)
            # print(length)
            # clicking mouse if distance is short
            if length<40:
                cv2.circle(img,(lineinfo[4],lineinfo[5]),15,(255,255,0),cv2.FILLED)
                pa.click()

    # frame rate
    ctime=time.time() 
    fps=1/(ctime-ptime) 
    ptime=ctime

    cv2.putText(img, str(int(fps)),(20,50), cv2.FONT_ITALIC, 1,(0,0,255),3)


    # display
    cv2.imshow("Image",img)
    cv2.waitKey(1)
