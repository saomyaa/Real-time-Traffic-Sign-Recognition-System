# Real Time Traffic Sign Recognition System
* Real time video is taken as an input for the system.
* Multiple frames from the video are captured until the camera is stopped.
* To specify redness in an image, a defined returnRedness() named function is used which converts 
the frame’s RGB color space to YUV, splits and returns v. cvtColor() is a predefined function in 
OpenCV which is used for this purpose.
* Applying binary thresholding to separate the traffic signs from the background noise. In binary 
thresholding, a threshold value is chosen, and each pixel in the image is compared against this 
threshold. If the pixel intensity is above the threshold, the pixel is assigned a value of 255 (white), 
indicating foreground or object of interest. If the pixel intensity is below the threshold, the pixel is 
assigned a value of 0 (black), representing background or irrelevant pixels.
* Applying morphology takes an input image and performs a closing morphological operation on it 
using a kernel of size determined by kernelSize. The closing operation combines dilation followed 
by erosion, useful for closing small holes or gaps in foreground objects. The function returns the 
resulting image after the morphological operation.
* The findContour function utilizes OpenCV's findContours function to extract contours from a binary 
image (img). The function returns the list of contours found in the image.
* The findBiggestContour function takes a list of contours and calculates the area of each contour. It 
then returns the contour with the largest area, which represents the biggest object in the image.
* Developed tensorflow and keras classifier model for the detection and recognition of the traffic 
signs.
* Defined a boundarybox function to highlight the position and size of the identified traffic sign 
within the video frame.
* Whenever, a traffic sign is recognized with the help of frames sent to model the system beeps with a 
frequency of 2500Hz and interval of 1 second as an audio feedback and the name of predicted sign 
is displayed in terminal.
* Using twilio we create a notification alert which is sent via SMS on driver’s phone number.
# How to Run
* Final.py is the main file of the project, to run type python Final.py in the terminal.
* Twilio account should be created if notification is a requirement and the SID, Token, Twilio account number and phone number should be replaced in the code. 
