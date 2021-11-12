import cv2
import imutils
import numpy as np

bg = None

# calculates the average of the background
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

if __name__ == "__main__":

    weight = 0.5
    camera = cv2.VideoCapture(0)
    frameCnt = 0 

    print("Starting...")

    while(True):
        
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        cp = frame.copy()

        gs = cv2.cvtColor(cp,cv2.COLOR_BGR2GRAY)
        gs = cv2.GaussianBlur(gs, (7, 7), 0)

        if (frameCnt < 30):
            bg_avg(gs, weight)
            print("BG sampled")
        else:
            hand = segment(gs)

            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(cp, segmented, -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
        cv2.imshow("Video", cp)
        frameCnt += 1
        input = cv2.waitKey(1) & 0xFF
        if input == ord("q"):
            break
    
    camera.release()
    cv2.destroyAllWindows()