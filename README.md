# Gesture-Recognition
This code shall recognize gestures with the help of OpenCV
NOTE: closing openCV windows will just spawn them again. To quit the script, either press 'q' while an openCV window is focussed or kill the process from the terminal

## Files
### segmentation.py
This script was the first attempt of using openCV to segment the Hand from the background and recognize gestures, before moving to the mediapipe library
- Required libraries:
  - cv2
  - imutils
  - numpy
### hand_detection.py
This script detects a hand by using the mediapipe library. The gesture recognizing function still needs to be implemented
- Required libraries:
  - cv2
  - imutils
  - mediapipe
