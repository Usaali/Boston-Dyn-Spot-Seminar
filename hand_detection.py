import cv2
import mediapipe as mp
import imutils
import time

def findHand(frame, handDetector):
    """ This function uses the mediapipe to detect a hand

    Args:
        frame: The image in which the hand should be detected
        handDetector: The mediapipe detector object

    Returns:
        An array with landmarks of the detected hand or None if nothing is detected.
        Landmarks are indexed by:
        WRIST = 0
        THUMB_CMC = 1
        THUMB_MCP = 2
        THUMB_IP = 3
        THUMB_TIP = 4
        INDEX_FINGER_MCP = 5
        INDEX_FINGER_PIP = 6
        INDEX_FINGER_DIP = 7
        INDEX_FINGER_TIP = 8
        MIDDLE_FINGER_MCP = 9
        MIDDLE_FINGER_PIP = 10
        MIDDLE_FINGER_DIP = 11
        MIDDLE_FINGER_TIP = 12
        RING_FINGER_MCP = 13
        RING_FINGER_PIP = 14
        RING_FINGER_DIP = 15
        RING_FINGER_TIP = 16
        PINKY_MCP = 17
        PINKY_PIP = 18
        PINKY_DIP = 19
        PINKY_TIP = 20
    """
    global mpDraw
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert input image to RGB
    results = handDetector.process(imgRGB) #run the image through the mediapipe detector

    if results.multi_hand_landmarks is not None:
        mpDraw.draw_landmarks(frame,results.multi_hand_landmarks[0],mpHands.HAND_CONNECTIONS) #draw the hand connections onto the frame. NOTE: comment this out if it is unwanted
        return results.multi_hand_landmarks[0].landmark
    else:
        return None

def contourHand(img):
    """ This function tries to process an image and find the contours of the hand

    Args:
        img: The image of the hand (and just the hand)

    Returns:
        A contour array
    """
    if(img is None):
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to gray for better thresholding
    gray = cv2.GaussianBlur(gray,(5,5),0) #add blur to reduce noise impact
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1] #threshold the image. NOTE: the threshold value is entirely experimental and varies with different lighting
    thresh = cv2.erode(thresh, None, iterations=2)  #add erosion effect to reduce noise impact
    thresh = cv2.dilate(thresh, None, iterations=2) #add dilation effect to reduce noise impact
    cv2.imshow("Tresholded",thresh) #show the tresholded image
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #find the contours
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(img, [c], -1, (0, 255, 255), 2) #draw the contours onto the image
    return c

def contourOnLine(x1,y1,x2,y2,c):
    """ This is a helper function to detect if a contour cuts a line between two points p1 and p2

    Args:
        x1: X value of p1
        y1: Y value of p1
        x2: X value of p2
        y2: Y value of p2
        c: The contour

    Returns:
        True if contour cuts line, False if it doesn't 
    """
    # Not sure if this works yet so its TODO!
    if(x1==x2 or y1==y2):
        return False
    m = (y2-y1)/(x2-x1) 
    for p in c:
        x3,y3=p[0]
        if min(x1,x2)<=x3<=max(x1,x2) and min(y1,y2)<=y3<=max(y1,y2):
            if y3-y1 == m * (x3-x1):
                return True
    return False

def fingersClosed(frame, landmarks, contour):
    """ This function tries to determine if the fingers of a hand are closed or not

    Args:
        frame: image to check
        landmarks: hand landmarks generated from mediapipe
        contour: the contour of the hand 

    Returns:
        True if they are closed, False if not
    """
    # This function does not work yet and is TODO!
    # NOTE: probably not working because contour is done on a subpicture and not on the whole image thus contour coordinates != frame coordinates?
    
    h, w = frame.shape[:2] #get frame height and width because landmarks give relative position
    for i in range(3):
        # use finger DIP landmarks to check
        p1 = landmarks[4*(1+i)-1] 
        p2 = landmarks[4*(2+i)-1]
        if contourOnLine(int(p1.x*w),int(p1.y*h),int(p2.x*w),int(p2.y*h),contour):
            return False
    return True

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

if __name__ == "__main__":
    cam = cv2.VideoCapture(0) #set default video input as camera

    #initialize the detector in video mode, with a maximum of one hand
    handDetector =  mpHands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    
    #variables for fps calculation
    pTime = 0
    cTime = 0
    
    while(True):
        (grabbed, frame) = cam.read() #get live image from camera
        frame = cv2.flip(frame, 1) #flip the image to get mirrored view

        landmarks = findHand(frame, handDetector) #let mediapipe detect the hand
        
        if landmarks is not None:
            minX, minY = frame.shape[:2]
            maxX = 0
            maxY = 0
            for id, lm in enumerate(landmarks):
                h, w, c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h) #get the absolute landmark coordinates
                #calculate min and max values for x and y to put a rectangle around the hand
                minX = min(minX,cx)
                maxX = max(maxX,cx)
                minY = min(minY,cy)
                maxY = max(maxY,cy)
                cv2.circle(frame, (cx,cy),3,(255,0,0),cv2.FILLED) #draw on every landmark
            cv2.rectangle(frame,(minX,minY),(maxX,maxY),(255,255,0),2) #put a rectangle around the hand
            margin = 20
            handPic = frame[minY-margin:maxY+margin,minX-margin:maxX+margin] #take an image from the hand with a margin because the landmark is not on the edge of a finger
            contour = contourHand(handPic)
            wrist = landmarks[0] #grab wrist position for placing text there 
            cv2.putText(frame,str(fingersClosed(frame,landmarks,contour)), (int(wrist.x*w),int(wrist.y*h)), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3) #check if fingers are closed and write the result onto the wrist
            cv2.imshow("Hand", handPic) #display the picture of the hand with contour
        #calculate fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) #print fps on frame

        cv2.imshow("Video", frame) #display live feed

        input = cv2.waitKey(1) & 0xFF #quit the program with q
        if input == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()