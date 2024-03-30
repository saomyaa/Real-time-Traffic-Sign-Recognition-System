import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
modelPath = 'C:\\Users\\91922\\Desktop\\BTP\\Traffic-Sign-Recognition-master\\03-Classification\\Models'
model = keras.models.load_model(modelPath+'/TSModel5')

def returnRedness(img):
	yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y,u,v=cv2.split(yuv)
	return v

def threshold(img,T=150):
	_,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img 

def findContour(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
	m = 0
	c = [cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]

def boundaryBox(img,contours):
	x,y,w,h=cv2.boundingRect(contours)
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	sign=img[y:(y+h) , x:(x+w)]
	return img,sign

def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

def predict4(sign):
	img=preprocessingImageToClassifier(sign,imageSize=28)
	return np.argmax(model.predict(img))


#--------------------------------------------------------------------------
labelToText={0:"Stop",
    		1:"Do not Enter",
    		2:"Traffic jam is close",
    		3:"Yeild"}
cap=cv2.VideoCapture(0)

while(True):
	_, frame = cap.read()
	# redness = returnRedness(frame) # step 1 --> specify the redness of the image
	# thresh = threshold(redness) 	
	# try:
	# 	contours = findContour(thresh)
	# 	big = findBiggestContour(contours)
	# 	if cv2.contourArea(big) > 3000:
	# 		print(cv2.contourArea(big))
	# 		img,sign = boundaryBox(frame,big)
	# 		cv2.imshow('frame',img)
	# 		cv2.imshow('sign',sign)
	# 	else:
	# 		cv2.imshow('frame',frame)
	# except:
	# 	cv2.imshow('frame',frame)

	
	# img = returnRedness(frame)
	# img = threshold(img,T = 150)
	# cv2.imshow('frame',img)
	# contours = findContour(img)
	# big = findBiggestContour(contours)
	# print(contours)
	# if cv2.contourArea(big) > 100:
	# 	print("Iam here")
	# 	img,sign = boundaryBox(frame,big)
	# 	cv2.imshow('frame',frame)
	# 	tic=time.time()
	# 	#print("Now,I see:",labelToText[predict4(sign)])
	# 	toc=time.time()
	# 	print("Running Time of Model4",(toc-tic)*1000,'ms')
	# else:	
	# 	cv2.imshow('frame',frame)
	
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break
	img=returnRedness(frame)
	img=threshold(img,T=150)
	try:
		contours=findContour(img)
		big=findBiggestContour(contours)
		if cv2.contourArea(big)>100:
			img,sign=boundaryBox(frame,big)
			cv2.imshow('frame',frame)
			tic=time.time()
			print("Now,I see:",labelToText[predict4(sign)])
			toc=time.time()
			print("Running Time of Model4",(toc-tic)*1000,'ms')
		else:
			cv2.imshow("frame",frame)
	except:
		cv2.imshow('frame',frame)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break
cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# # modelPath = r'C:\Users\91922\Desktop\BTP\Traffic-Sign-Recognition-master\03-Classification\Models'
# model = keras.models.load_model(r'C:\Users\91922\Desktop\BTP\Traffic-Sign-Recognition-master\03-Classification\Models\TSModel5')

# def returnRedness(img):
# 	yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
# 	y, u, v = cv2.split(yuv)
# 	return v

# def threshold(img,T=150):
# 	_, img = cv2.threshold(img,T,255,cv2.THRESH_BINARY)
# 	return img 

# def findContour(img):
# 	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 	return contours

# def findBiggestContour(contours):
# 	m = 0
# 	c = [cv2.contourArea(i) for i in contours]
# 	return contours[c.index(max(c))]

# def boundaryBox(img,contours):
# 	x, y, w, h = cv2.boundingRect(contours)
# 	img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
# 	sign = img[y:(y+h) , x:(x+w)]
# 	return img, sign

# def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
#     image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#     image = cv2.resize(image,(imageSize,imageSize))
#     image = (image - mu) / std
#     image = image.reshape(1,imageSize,imageSize,1)
#     return image

# def predict(sign):
# 	img = preprocessingImageToClassifier(sign,imageSize=28)
# 	return np.argmax(model.predict(img))

# #--------------------------------------------------------------------------
# labelToText = { 0:"Stop",
#     			1:"Do not Enter",
#     			2:"Traffic jam is close",
#     			3:"Yeild"}
# cap=cv2.VideoCapture(0)

# while(True):
# 	_, frame = cap.read()
# 	redness = returnRedness(frame) # step 1 --> specify the redness of the image
# 	thresh = threshold(redness) 	
# 	try:
# 		contours = findContour(thresh)
# 		big = findBiggestContour(contours)
# 		if cv2.contourArea(big) > 3000:
# 			print(cv2.contourArea(big))
# 			img,sign = boundaryBox(frame,big)
# 			cv2.imshow('frame',img)
# 			print("Now,I see:",labelToText[predict(sign)])

# 		else:
# 			cv2.imshow('frame',frame)
# 	except:
# 		cv2.imshow('frame',frame)

# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

# cap.release()
# cv2.destroyAllWindows()









# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import winsound

# # Load the trained model
# model = keras.models.load_model(r'C:\Users\91922\Desktop\BTP\Traffic-Sign-Recognition-master\03-Classification\Models\TSModel5')

# # Function to return the redness of the image
# def returnRedness(img):
#     yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     y, u, v = cv2.split(yuv)
#     return v

# # Function to apply thresholding to the image
# def threshold(img, T=150):
#     _, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
#     return img 

# # Function to find contours in the image
# def findContour(img):
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# # Function to find the biggest contour in the image
# def findBiggestContour(contours):
#     c = [cv2.contourArea(i) for i in contours]
#     return contours[c.index(max(c))]

# # Function to draw boundary box around the detected sign and label it
# def boundaryBox(img, contours, label):
#     x, y, w, h = cv2.boundingRect(contours)
#     img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
#     cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     sign = img[y:(y+h), x:(x+w)]
#     return img, sign

# # Function to preprocess image for classification
# def preprocessingImageToClassifier(image=None, imageSize=28, mu=89.77428691773054, std=70.85156431910688):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Changed from RGB to BGR
#     image = cv2.resize(image, (imageSize, imageSize))
#     image = (image - mu) / std
#     image = image.reshape(1, imageSize, imageSize, 1)
#     return image

# # Function to predict the traffic sign label
# def predict(sign):
#     img = preprocessingImageToClassifier(sign, imageSize=28)
#     return np.argmax(model.predict(img))

# # Dictionary mapping label indices to text descriptions
# labelToText = {
#     0: "Stop",
#     1: "Do not Enter",
#     2: "Traffic jam is close",
#     3: "Yeild"
# }

# # Main code
# cap = cv2.VideoCapture(0)

# beep_played = False  # Variable to keep track of whether beep sound has been played
# beep_time = None     # Variable to store the time when the last beep was played

# while True:
#     _, frame = cap.read()
#     redness = returnRedness(frame)
#     thresh = threshold(redness) 

#     try:
#         contours = findContour(thresh)
#         big = findBiggestContour(contours)

#         if cv2.contourArea(big) > 3000:
#             img, sign = boundaryBox(frame, big,labelToText[predict(sign)])
#             cv2.imshow('frame', img)
#             current_time = cv2.getTickCount()
#             if (not beep_played) or (current_time - beep_time > 5000):  # Play beep sound if not played or 5 seconds have passed since the last beep
#                 label = predict(sign)
#                 print("Now, I see:", labelToText[label])
#                 winsound.Beep(2500, 1000)  # Beep sound
#                 beep_played = True
#                 beep_time = current_time
#         else:
#             cv2.imshow('frame', frame)
#     except:
#         cv2.imshow('frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

