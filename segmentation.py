import cv2
import imutils
import numpy as np

bg = None   # variable to store the background

def bg_avg(img, weight):
    """ This function takes an image and adds its weight to the averaged background variable

    Args:
        img: the image that should be processed
        weight: the weight with which the image impacts the background
    """
    global bg

    #initialize the background on first run
    if bg is None:
        bg = img.copy().astype("float")
        return

    #get the average weight of the image
    cv2.accumulateWeighted(img,bg,weight)    

#segments a region from the background
def segment(img, threshold=25):
    """ This function takes an image, calculates the difference from the background variable and returns a contour of the difference

    Args:
        img: The image that is segmented
        threshold: The threshold that is used to segment the image. Defaults to 25.

    Returns:
        A tuple containing the thresholded image and the contour
    """
    global bg
    #calculate absolute difference between input and background
    diff = cv2.absdiff(bg.astype("uint8"),img)
    #threshold the result to get a clearer image
    thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)[1]
    #get contours of the difference
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(cnts))  == 0: #no contour was found (something went wrong)
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

if __name__ == "__main__":

    camera = cv2.VideoCapture(0) #use the standard video capture device as input
    frameCnt = 0 #a counter to count background frames

    print("Starting...")

    while(True):
        
        (grabbed, frame) = camera.read() #grab a live camera image

        frame = imutils.resize(frame, width=700) 
        frame = cv2.flip(frame, 1) #flip image so that its mirrored correctly
        cp = frame.copy()

        gs = cv2.cvtColor(cp,cv2.COLOR_BGR2GRAY) #convert the image to grayscale
        gs = cv2.GaussianBlur(gs, (7, 7), 0) #blur the image to reduce noisy areas

        if (frameCnt < 30): #sample the initial 30 frames for the background (NOTE: Camera must not be moved during this!)
            bg_avg(gs, 0.5)
            print("BG sampled")
        else:
            hand = segment(gs) #get the segmented picture

            #display the thresholded image and draw the contours onto live feed
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(cp, segmented, -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
        cv2.imshow("Video", cp) #display live feed
        frameCnt += 1
        input = cv2.waitKey(1) & 0xFF
        if input == ord("q"): #pressing q will quit the program
            break
    
    camera.release()
    cv2.destroyAllWindows()