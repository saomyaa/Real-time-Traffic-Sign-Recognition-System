import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import winsound
from twilio.rest import Client

# Load the trained model
model = keras.models.load_model(r'C:\Users\91922\Desktop\BTP\Traffic-Sign-Recognition-master\03-Classification\Models\TSModel5')

# Function to return the redness of the image
def returnRedness(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return v

# Function to apply thresholding to the image
def threshold(img, T=150):
    _, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return img 

# Function to find contours in the image
def findContour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to find the biggest contour in the image
def findBiggestContour(contours):
    c = [cv2.contourArea(i) for i in contours]
    return contours[c.index(max(c))]

# Function to draw boundary box around the detected sign
def boundaryBox(img, contours):
    x, y, w, h = cv2.boundingRect(contours)
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    sign = img[y:(y+h), x:(x+w)]
    return img, sign

# Function to preprocess image for classification
def preprocessingImageToClassifier(image=None, imageSize=28, mu=89.77428691773054, std=70.85156431910688):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (imageSize, imageSize))
    image = (image - mu) / std
    image = image.reshape(1, imageSize, imageSize, 1)
    return image

# Function to predict the traffic sign label
def predict(sign):
    img = preprocessingImageToClassifier(sign, imageSize=28)
    return np.argmax(model.predict(img))

# Dictionary mapping label indices to text descriptions
labelToText = {
    0: "Stop",
    1: "Do not Enter",
    2: "Traffic jam is close",
    3: "Yeild"
}

def access_webcam():
    st.title("Webcam Access")
    st.write("Here you can access the webcam.")

    cap = cv2.VideoCapture(0)

    beep_played = False  # Variable to keep track of whether beep sound has been played
    beep_time = None     # Variable to store the time when the last beep was played

    # Display start/stop buttons
    start_stop_container = st.container()
    # start_button = start_stop_container.button("Start Webcam")
    stop_button = start_stop_container.button("Stop Webcam")

    webcam_output = st.empty()

    while True:
        _, frame = cap.read()
        redness = returnRedness(frame)
        thresh = threshold(redness)

        try:
            contours = findContour(thresh)
            big = findBiggestContour(contours)

            if cv2.contourArea(big) > 3000:
                img, sign = boundaryBox(frame, big)
                cv2.imshow('frame', img)
                current_time = cv2.getTickCount()

                if (not beep_played) or (current_time - beep_time > 5000):
                    label = predict(sign)
                    print("Now, I see:", labelToText[label])
                    winsound.Beep(2500, 1000)

                    beep_played = True
                    beep_time = current_time
                    #create a twilio account 
                    account_sid = '###'
                    auth_token = '###'
                    #replace '##' with the actual SID and TOKEN of ur account 
                    client = Client(account_sid, auth_token)

                    message = client.messages.create(
                    from_='***',
                    #replace with your twilio phone number
                    body='signallll............... ',
                    to='**'
                    #replace it with your phone number where you want a notification
                    )

                    st.balloons()

            else:
                cv2.imshow('frame', frame)

        except:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if stop_button:
            cap.release()
            cv2.destroyAllWindows()
            break

        # Update Streamlit display
        webcam_output.image(frame, channels="BGR")

    # Cleanup after stopping webcam
    cv2.destroyAllWindows()

def main():
    st.set_page_config(page_title="Traffic Sign Recognition Project", page_icon=":traffic_light:")

    # Streamlit theme customization
    st.markdown(
        """
        <style>
            body {
                background-color: #1e272e;
                color: #ecf0f1;
            }
            .streamlit-container {
                background-color: #192a56;
                padding-top: 1rem;
                padding-bottom: 1rem;
                border-radius: 0.5rem;
            }
            .streamlit-button {
                background-color: #3498db;
                color: #ecf0f1;
            }
            .streamlit-button:hover {
                background-color: #2980b9;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Traffic Sign Recognition and Alerts")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image('Picture2.png')

    st.write("How it works:")
    st.write("1. The webcam captures video frames.")
    st.write("2. The redness of the frames is analyzed to identify potential traffic signs.")
    st.write("3. If a significant contour is detected, the system predicts the traffic sign label.")
    st.write("4. The label is displayed dynamically on the Streamlit page and logged.")
    st.write("5. A beep sound is played, and a Twilio message is sent with the detected sign.")
    st.write("Detected Traffic Signs:")
    st.write("The detected traffic sign will be displayed dynamically here.")


    if st.button("Access Webcam"):
        access_webcam()

if __name__ == '__main__':
    main()