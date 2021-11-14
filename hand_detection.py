import cv2
import mediapipe as mp
import imutils
import time

def bg_avg(img, weight):
    global bg

    #initialize the background on first run
    if bg is None:
        bg = img.copy().astype("float")
        return
    #get the average weight of the image
    cv2.accumulateWeighted(img,bg,weight)    

#segments a region from the background
def segment(img, threshold=25):
    global bg
    #calculate absolute difference between input and background
    diff = cv2.absdiff(bg.astype("uint8"),img)
    #threshold the result to get a clearer image
    thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)[1]
    #get contours of the difference
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(cnts))  == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def findHand(frame, handDetector):
    global mpDraw
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = handDetector.process(imgRGB)
    if results.multi_hand_landmarks is not None:
        mpDraw.draw_landmarks(frame,results.multi_hand_landmarks[0],mpHands.HAND_CONNECTIONS)
        return results.multi_hand_landmarks[0].landmark
    else:
        return None

def contourHand(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
       
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(img, [c], -1, (0, 255, 255), 2)

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    handDetector =  mpHands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

    pTime = 0
    cTime = 0
    while(True):
        (grabbed, frame) = cam.read()
        #frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        landmarks = findHand(frame,handDetector)
        #landmarks = None
        if landmarks is not None:
            minX, minY = frame.shape[:2]
            maxX = 0
            maxY = 0
            for lm in landmarks:
                h, w, c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                minX = min(minX,cx)
                maxX = max(maxX,cx)
                minY = min(minY,cy)
                maxY = max(maxY,cy)
                cv2.circle(frame, (cx,cy),3,(255,0,0),cv2.FILLED)
            cv2.rectangle(frame,(minX,minY),(maxX,maxY),(255,255,0),2)
            margin = 20
            handPic = frame[minY-margin:maxY+margin,minX-margin:maxX+margin]
            contourHand(handPic)
            cv2.imshow("Hand", handPic)
            #indexFinger = landmarks[0]
            #cv2.putText(frame,results.multi_handedness[0].classification[0].label, (int(indexFinger.x*w),int(indexFinger.y*h)), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        #cv2.imshow("Gauss",gray)
        #cv2.imshow("Tresh",thresh)

        cv2.imshow("Video",frame)
        input = cv2.waitKey(1) & 0xFF
        if input == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()